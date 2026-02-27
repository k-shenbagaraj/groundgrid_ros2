#include <groundgrid/GroundSegmentation.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <chrono>
#include <unordered_map>
#include <algorithm>
#include <thread>
#include <cmath>
#include <array>

using namespace groundgrid;

void GroundSegmentation::init(const size_t dimension, const float& resolution) {
    const size_t cellCount = std::round(dimension / resolution);

    expectedPoints.resize(cellCount, cellCount);
    for (size_t i = 0; i < cellCount; ++i) {
        for (size_t j = 0; j < cellCount; ++j) {
            const float& dist = std::hypot(i - cellCount / 2.0, j - cellCount / 2.0);
            expectedPoints(i, j) = std::atan(1 / dist) / verticalPointAngDist;
        }
    }

    Eigen::initParallel();
}

pcl::PointCloud<GroundSegmentation::PCLPoint>::Ptr GroundSegmentation::filter_cloud(
    const pcl::PointCloud<PCLPoint>::Ptr cloud, 
    const PCLPoint& cloudOrigin, 
    const geometry_msgs::msg::TransformStamped& mapToBase, 
    grid_map::GridMap& map)
{
    auto start = std::chrono::steady_clock::now();
    static double avg_insertion_time = 0.0;
    static double avg_detection_time = 0.0;
    static double avg_segmentation_time = 0.0;
    static unsigned int time_vals = 0;

    pcl::PointCloud<PCLPoint>::Ptr filtered_cloud(new pcl::PointCloud<PCLPoint>);
    filtered_cloud->points.reserve(cloud->points.size());

    map.add("groundCandidates", 0.0);
    map.add("planeDist", 0.0);
    map.add("m2", 0.0);
    map.add("meanVariance", 0.0);

    // raw point count layer for the evaluation
    map.add("pointsRaw", 0.0);

    map["groundCandidates"].setZero();
    map["points"].setZero();
    map["minGroundHeight"].setConstant(std::numeric_limits<float>::max());
    map["maxGroundHeight"].setConstant(std::numeric_limits<float>::min());

    map.add("variance", 0.0);
    static const grid_map::Matrix& ggv = map["variance"];
    static grid_map::Matrix& gpl = map["points"];
    static grid_map::Matrix& ggl = map["ground"];
    const auto& size = map.getSize();
    const size_t threadcount = mConfig.thread_count;

    std::vector<std::pair<size_t, grid_map::Index>> point_index;
    point_index.reserve(cloud->points.size());
    std::vector<std::vector<std::pair<size_t, grid_map::Index>>> point_index_list;
    point_index_list.resize(threadcount);

    // Collect all outliers for the outlier detection evaluation
    std::vector<size_t> outliers;
    std::vector<std::vector<size_t>> outliers_list;
    outliers_list.resize(threadcount);

    // store ignored points to re-add them afterwards
    std::vector<std::pair<size_t, grid_map::Index>> ignored;
    std::vector<std::vector<std::pair<size_t, grid_map::Index>>> ignored_list;
    ignored_list.resize(threadcount);

    // Divide the point cloud into threadcount sections for threaded calculations
    std::vector<std::thread> threads;

    for (size_t i = 0; i < threadcount; ++i) {
        const size_t start = std::floor((i * cloud->points.size()) / threadcount);
        const size_t end = std::ceil(((i + 1) * cloud->points.size()) / threadcount);
        threads.push_back(std::thread(&GroundSegmentation::insert_cloud, this, cloud, start, end, std::cref(cloudOrigin), 
                            std::ref(point_index_list[i]), std::ref(ignored_list[i]), std::ref(outliers_list[i]), std::ref(map)));
    }

    // Wait for results
    std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));

    // Join results
    for (const auto& point_index_part : point_index_list)
        point_index.insert(point_index.end(), point_index_part.begin(), point_index_part.end());
    for (const auto& outlier_index_part : outliers_list)
        outliers.insert(outliers.end(), outlier_index_part.begin(), outlier_index_part.end());
    for (const auto& ignored_part : ignored_list)
        ignored.insert(ignored.end(), ignored_part.begin(), ignored_part.end());

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    const double milliseconds = elapsed_seconds.count() * 1000;
    avg_insertion_time = (milliseconds + time_vals * avg_insertion_time) / (time_vals + 1);
    RCLCPP_DEBUG(rclcpp::get_logger("GroundSegmentation"), "ground point rasterization took %f ms (avg %f ms)", milliseconds, avg_insertion_time);

    start = std::chrono::steady_clock::now();

    // Divide the grid map into four sections for threaded calculations
    threads.clear();
    for (unsigned short section = 0; section < 4; ++section)
        threads.push_back(std::thread(&GroundSegmentation::detect_ground_patches, this, std::ref(map), section));

    // Wait for results
    std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));
    end = std::chrono::steady_clock::now();
    elapsed_seconds = end - start;
    avg_detection_time = (elapsed_seconds.count() * 1000 + time_vals * avg_detection_time) / (time_vals + 1);
    RCLCPP_DEBUG(rclcpp::get_logger("GroundSegmentation"), "ground patch detection took %ld ms (avg %f ms)",
                std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(), avg_detection_time);
    ++time_vals;

    start = std::chrono::steady_clock::now();
    spiral_ground_interpolation(map, mapToBase);
    end = std::chrono::steady_clock::now();
    RCLCPP_DEBUG(rclcpp::get_logger("GroundSegmentation"), "ground interpolation took %ld ms",
                std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    start = std::chrono::steady_clock::now();
    update_hole_cost_layer(map);
    map["points"].setConstant(0.0);

    // Re-add ignored points
    point_index.insert(point_index.end(), ignored.begin(), ignored.end());

    // Debugging statistics
    const double& min_dist_fac = mConfig.minimum_distance_factor * 5;
    const double& min_point_height_thres = mConfig.miminum_point_height_threshold;
    const double& min_point_height_obs_thres = mConfig.minimum_point_height_obstacle_threshold;

    for (const std::pair<size_t, grid_map::Index>& entry : point_index) {
        const PCLPoint& point = cloud->points[entry.first];
        const grid_map::Index& gi = entry.second;
        const double& groundheight = ggl(gi(0), gi(1));

        // copy the points intensity because it gets overwritten for evaluation purposes
        const float& variance = ggv(gi(0), gi(1));

        if (size(0) <= gi(0) + 3 || size(1) <= gi(1) + 3)
            continue;

        const float dist = std::hypot(point.x - cloudOrigin.x, point.y - cloudOrigin.y);
        const double tolerance = std::max(std::min((min_dist_fac * dist) / variance * min_point_height_thres, min_point_height_thres), min_point_height_obs_thres);

        if (tolerance + groundheight < point.z) { // non-ground points
            PCLPoint& segmented_point = filtered_cloud->points.emplace_back(point);
            segmented_point.intensity = 99;
            gpl(gi(0), gi(1)) += 1.0f;
        } else { // ground point
            PCLPoint& segmented_point = filtered_cloud->points.emplace_back(point);
            segmented_point.intensity = 49;
        }
    }

    // Re-add outliers to cloud
    for (size_t i : outliers) {
        const PCLPoint& point = cloud->points[i];
        PCLPoint& segmented_point = filtered_cloud->points.emplace_back(point); // ground point
        segmented_point.intensity = 49;
    }

    end = std::chrono::steady_clock::now();
    elapsed_seconds = end - start;
    avg_segmentation_time = (elapsed_seconds.count() * 1000 + (time_vals - 1) * avg_segmentation_time) / time_vals;
    RCLCPP_DEBUG(rclcpp::get_logger("GroundSegmentation"), "point cloud segmentation took %ld ms (avg %f ms)",
                std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(), avg_segmentation_time);

    return filtered_cloud;
}

void GroundSegmentation::insert_cloud(const pcl::PointCloud<PCLPoint>::Ptr cloud, const size_t start, const size_t end, const PCLPoint& cloudOrigin, 
    std::vector<std::pair<size_t, grid_map::Index>>& point_index,
    std::vector<std::pair<size_t, grid_map::Index>>& ignored, 
    std::vector<size_t>& outliers, grid_map::GridMap& map)
{
    static const grid_map::Matrix& ggp = map["groundpatch"];

    static grid_map::Matrix& gpr = map["pointsRaw"];
    static grid_map::Matrix& gpl = map["points"];
    static grid_map::Matrix& ggl = map["ground"];
    static grid_map::Matrix& gmg = map["groundCandidates"];
    static grid_map::Matrix& gmm = map["meanVariance"];
    static grid_map::Matrix& gmx = map["maxGroundHeight"];
    static grid_map::Matrix& gmi = map["minGroundHeight"];
    static grid_map::Matrix& gmd = map["planeDist"];
    static grid_map::Matrix& gm2 = map["m2"];

    const auto& size = map.getSize();

    point_index.reserve(end-start);

    for (size_t i = start; i < end; ++i)
    {
        const PCLPoint& point = cloud->points[i];
        const auto& pos = grid_map::Position(point.x, point.y);
        const float sqdist = std::pow(point.x - cloudOrigin.x, 2.0) + std::pow(point.y - cloudOrigin.y, 2.0);

        bool toSkip = false;

        grid_map::Index gi;
        map.getIndex(pos, gi);

        if (!map.isInside(pos))
            continue;

        // point count map used for evaluation
        gpr(gi(0), gi(1)) += 1.0f;

        if (point.ring > mConfig.max_ring || sqdist < minDistSquared) {
            ignored.push_back(std::make_pair(i, gi));
            continue;
        }

        // Outlier detection test
        const float oldgroundheight = ggl(gi(0), gi(1));
        if (point.z < oldgroundheight - 0.2) {
            // get direction
            PCLPoint vec;
            vec.x = point.x - cloudOrigin.x;
            vec.y = point.y - cloudOrigin.y;
            vec.z = point.z - cloudOrigin.z;

            float len = std::sqrt(std::pow(vec.x, 2.0f) + std::pow(vec.y, 2.0f) + std::pow(vec.z, 2.0f));
            vec.x /= len;
            vec.y /= len;
            vec.z /= len;

            // check for occlusion
            for (int step = 3; (std::pow(step * vec.x, 2.0) + std::pow(step * vec.y, 2.0) + std::pow(step * vec.z, 2.0)) < std::pow(len, 2.0) && vec.z < -0.01f; ++step) {
                grid_map::Index intersection, pointPosIndex;
                grid_map::Position intersecPos(step * (vec.x) + cloudOrigin.x, step * (vec.y) + cloudOrigin.y);
                map.getIndex(intersecPos, intersection);

                // Check if inside map borders
                if (intersection(0) <= 0 || intersection(1) <= 0 || intersection(0) >= size(0) - 1 || intersection(1) >= size(1) - 1)
                    continue;

                // check if known ground occludes the line of sight
                const auto& block = ggp.block<3, 3>(std::max(intersection(0) - 1, 2), std::max(intersection(1) - 1, 2));
                if (block.sum() > mConfig.min_outlier_detection_ground_confidence && ggp(intersection(0), intersection(1)) > 0.01f && ggl(intersection(0), intersection(1)) >= step * vec.z + cloudOrigin.z + mConfig.outlier_tolerance) {
                    outliers.push_back(i);
                    toSkip = true;
                    break;
                }
            }
        }

        if (toSkip)
            continue;

        float& groundheight = gmg(gi(0), gi(1));
        float& mean = gmm(gi(0), gi(1));

        float planeDist = 0.0;
        point_index.push_back(std::make_pair(i, gi));

        float& points = gpl(gi(0), gi(1));
        float& maxHeight = gmx(gi(0), gi(1));
        float& minHeight = gmi(gi(0), gi(1));
        float& planeDistMap = gmd(gi(0), gi(1));
        float& m2 = gm2(gi(0), gi(1));

        planeDist = point.z - cloudOrigin.z;
        groundheight = (point.z + points * groundheight) / (points + 1.0);

        if (mean == 0.0)
            mean = planeDist;
        if (!std::isnan(planeDist)) {
            float delta = planeDist - mean;
            mean += delta / (points + 1);
            planeDistMap = (planeDist + points * planeDistMap) / (points + 1.0);
            m2 += delta * (planeDist - mean);
        }

        maxHeight = std::max(maxHeight, point.z);
        minHeight = std::min(minHeight, point.z - 0.0001f); // to make sure maxHeight > minHeight
        points += 1.0;
    }
}

void GroundSegmentation::detect_ground_patches(grid_map::GridMap& map, unsigned short section) const
{
    const grid_map::Matrix& gcl = map["groundCandidates"];
    const static auto& size = map.getSize();
    const static float resolution = map.getResolution();
    static const grid_map::Matrix& gm2 = map["m2"];
    static const grid_map::Matrix& gpl = map["points"];
    static grid_map::Matrix& ggv = map["variance"];
    
    // Calculate variance
    ggv = gm2.array().cwiseQuotient(gpl.array() + std::numeric_limits<float>::min());

    int cols_start = 2 + section % 2 * (gcl.cols() / 2 - 2);
    int rows_start = section >= 2 ? gcl.rows() / 2 : 2;
    int cols_end = (gcl.cols()) / 2 + section % 2 * (gcl.cols() / 2 - 2);
    int rows_end = section >= 2 ? gcl.rows() - 2 : (gcl.rows()) / 2;

    // Iterate through the sections
    for (int i = cols_start; i < cols_end; ++i) {
        for (int j = rows_start; j < rows_end; ++j) {
            const float sqdist = (std::pow(i - (size(0) / 2.0), 2.0) + std::pow(j - (size(1) / 2.0), 2.0)) * std::pow(resolution, 2.0);

            // Choose detection method based on distance
            if (sqdist <= std::pow(mConfig.patch_size_change_distance, 2.0)) {
                detect_ground_patch<3>(map, i, j);
            } else {
                detect_ground_patch<5>(map, i, j);
            }
        }
    }
}

template <int S>
void GroundSegmentation::detect_ground_patch(grid_map::GridMap& map, size_t i, size_t j) const
{
    static grid_map::Matrix& ggl = map["ground"];
    static grid_map::Matrix& ggp = map["groundpatch"];
    static grid_map::Matrix& ggv = map["variance"];
    static const grid_map::Matrix& gmi = map["minGroundHeight"];
    static const grid_map::Matrix& gpl = map["points"];
    static const auto& size = map.getSize();
    static const float resolution = map.getResolution();
    const int center_idx = std::floor(S / 2);

    const auto& pointsBlock = gpl.block<S, S>(i - center_idx, j - center_idx);
    const float sqdist = (std::pow(i - (size(0) / 2.0), 2.0) + std::pow(j - (size(1) / 2.0), 2.0)) * std::pow(resolution, 2.0);
    const int patchSize = S;
    const float& expectedPointCountperLaserperCell = expectedPoints(i, j);
    const float& pointsblockSum = pointsBlock.sum();
    float& oldConfidence = ggp(i, j);
    float& oldGroundheight = ggl(i, j);

    // Early skipping of (almost) empty areas
    if (pointsblockSum < std::max(std::floor(mConfig.ground_patch_detection_minimum_point_count_threshold * patchSize * expectedPointCountperLaserperCell), 3.0))
        return;

    // Calculation of variance threshold
    const float varThresholdsq = std::min(std::max(sqdist * std::pow(mConfig.distance_factor, 2.0), std::pow(mConfig.minimum_distance_factor, 2.0)), std::pow(mConfig.minimum_distance_factor * 10, 2.0));
    const auto& varblock = ggv.block<S, S>(i - center_idx, j - center_idx);
    const auto& minblock = gmi.block<S, S>(i - center_idx, j - center_idx);
    const float& variance = varblock(center_idx, center_idx);
    const float& localmin = minblock.minCoeff();
    const float maxVar = pointsBlock(center_idx, center_idx) >= mConfig.point_count_cell_variance_threshold ? variance : pointsBlock.array().cwiseProduct(varblock.array()).sum() / pointsblockSum;
    const float groundlevel = pointsBlock.cwiseProduct(minblock).sum() / pointsblockSum;
    const float groundDiff = std::max((groundlevel - oldGroundheight) * (2.0f * oldConfidence), 1.0f);

    // Do not update known high confidence estimations upward
    if (oldConfidence > 0.5 && groundlevel >= oldGroundheight + mConfig.outlier_tolerance) {
        RCLCPP_DEBUG(rclcpp::get_logger("GroundSegmentation"), "Skipping patch at (%zu, %zu), high confidence ground level detected.", i, j);
        return;
    }

    if (varThresholdsq > std::pow(maxVar, 2.0) && maxVar > 0 && pointsblockSum > (groundDiff * expectedPointCountperLaserperCell * patchSize) * mConfig.ground_patch_detection_minimum_point_count_threshold) {
        const float& newConfidence = std::min(pointsblockSum / mConfig.occupied_cells_point_count_factor, 1.0);
        // Calculate ground height
        oldGroundheight = (groundlevel * newConfidence + oldConfidence * oldGroundheight * 2) / (newConfidence + oldConfidence * 2);
        // Update confidence
        oldConfidence = std::min((pointsblockSum / (mConfig.occupied_cells_point_count_factor * 2.0f) + oldConfidence) / 2.0, 1.0);
    }
    else if (localmin < oldGroundheight) {
        // Update ground height
        oldGroundheight = localmin;
        // Update confidence
        oldConfidence = std::min(oldConfidence + 0.1f, 0.5f);
    }

    RCLCPP_DEBUG(rclcpp::get_logger("GroundSegmentation"), "Updated patch at (%zu, %zu), ground height: %f, confidence: %f", i, j, oldGroundheight, oldConfidence);
}

void GroundSegmentation::update_hole_cost_layer(grid_map::GridMap &map) const
{
    map.add("signed_height_residual", 0.0);
    map.add("hole_persistence", 0.0);
    map.add("hole_cost", 0.0);

    grid_map::Matrix &residual = map["signed_height_residual"];
    grid_map::Matrix &persistence = map["hole_persistence"];
    grid_map::Matrix &hole_cost = map["hole_cost"];

    const grid_map::Matrix &min_height = map["minGroundHeight"];
    const grid_map::Matrix &ground = map["ground"];
    const grid_map::Matrix &points = map["points"];

    const float strong_negative_threshold = -static_cast<float>(mConfig.hole_negative_residual_threshold);
    const float sigma = std::max(static_cast<float>(mConfig.hole_cost_sigma), 1e-4f);
    const float persistence_increase = static_cast<float>(mConfig.hole_persistence_increase);
    const float persistence_decay = static_cast<float>(mConfig.hole_persistence_decay);
    const float no_observation_decay = static_cast<float>(mConfig.hole_no_observation_decay);
    const auto residual_to_cost = [sigma](const float residual_delta) {
        const float exponent = std::clamp(residual_delta / sigma, -40.0f, 40.0f);
        return 1.0f / (1.0f + std::exp(exponent));
    };

    for (grid_map::GridMapIterator iterator(map); !iterator.isPastEnd(); ++iterator)
    {
        const grid_map::Index index(*iterator);
        const int i = index(0);
        const int j = index(1);

        float &residual_cell = residual(i, j);
        float &persistence_cell = persistence(i, j);
        float &hole_cost_cell = hole_cost(i, j);

        if (points(i, j) <= 0.0f)
        {
            persistence_cell = std::max(0.0f, persistence_cell - no_observation_decay);
            const float raw_cost = residual_to_cost(residual_cell);
            hole_cost_cell = persistence_cell * raw_cost;
            continue;
        }

        const float delta = min_height(i, j) - ground(i, j);
        residual_cell = delta;

        if (delta <= strong_negative_threshold)
        {
            persistence_cell = std::min(1.0f, persistence_cell + persistence_increase);
        }
        else
        {
            persistence_cell = std::max(0.0f, persistence_cell - persistence_decay);
        }

        // Sigmoid(-delta / sigma): large negative residuals map to high obstacle cost.
        const float raw_cost = residual_to_cost(delta);
        hole_cost_cell = persistence_cell * raw_cost;
    }
}

void GroundSegmentation::spiral_ground_interpolation(grid_map::GridMap &map, const geometry_msgs::msg::TransformStamped &toBase) const
{
    static grid_map::Matrix& ggl = map["ground"];
    static grid_map::Matrix& gvl = map["groundpatch"];
    const auto& map_size = map.getSize();
    const auto center_idx = map_size(0)/2 - 1;

    // Set initial values for the ground patch
    gvl(center_idx, center_idx) = 1.0f;
    
    geometry_msgs::msg::PointStamped ps;
    ps.header.frame_id = "base_link";
    
    // Transform the point to the appropriate frame
    try {
        tf2::doTransform(ps, ps, toBase);
    } catch (const tf2::TransformException &ex) {
        RCLCPP_ERROR(rclcpp::get_logger("GroundSegmentation"), "Transform failed: %s", ex.what());
        return;
    }

    // Set center to the current vehicle height
    ggl(center_idx, center_idx) = ps.point.z;

    // Iterate in a spiral pattern for interpolation
    for (int i = center_idx - 1; i >= 1; --i) {
        // Calculate the position of the rectangle (top left corner)
        int rectangle_pos = i;

        // Calculate the side length of the rectangle
        int side_length = (center_idx - rectangle_pos) * 2;

        // Interpolate the top and left sides
        for (short side = 0; side < 2; ++side) {
            for (int pos = rectangle_pos; pos < rectangle_pos + side_length; ++pos) {
                const int x = side % 2 ? pos : rectangle_pos;
                const int y = side % 2 ? rectangle_pos : pos;

                interpolate_cell(map, x, y);  // Assuming this function is already ROS 2 compatible
            }
        }

        // Interpolate the bottom and right sides
        rectangle_pos += side_length;
        for (short side = 0; side < 2; ++side) {
            for (int pos = rectangle_pos; pos >= rectangle_pos - side_length; --pos) {
                int x = side % 2 ? pos : rectangle_pos;
                int y = side % 2 ? rectangle_pos : pos;

                interpolate_cell(map, x, y);  // Assuming this function is already ROS 2 compatible
            }
        }
    }
}

void GroundSegmentation::interpolate_cell(grid_map::GridMap &map, const size_t x, const size_t y) const
{
    static const auto& center_idx = map.getSize()(0)/2 - 1;
    static grid_map::Matrix& gvl = map["groundpatch"];
    static grid_map::Matrix& ggl = map["ground"];
    constexpr float min_weight = 1e-6f;
    constexpr float min_gradient = 1e-4f;
    const int xi = static_cast<int>(x);
    const int yi = static_cast<int>(y);
    const float resolution = map.getResolution();

    float& height = ggl(x, y);
    float& occupied = gvl(x, y);

    const float min_confidence = std::max(static_cast<float>(mConfig.interpolation_min_confidence), 0.0f);
    const float sigma_height = std::max(static_cast<float>(mConfig.interpolation_height_sigma), 1e-4f);
    const float sigma_slope = std::max(static_cast<float>(mConfig.interpolation_slope_sigma), 1e-4f);
    const float anisotropic_factor = std::max(static_cast<float>(mConfig.interpolation_anisotropic_direction_factor), 0.0f);
    const float spatial_sigma = std::max(1.5f * resolution, 1e-4f);

    struct NeighborSample
    {
        int dx;
        int dy;
        float sample_height;
        float sample_confidence;
        float distance;
        float spatial_weight;
    };
    std::array<NeighborSample, 8> neighbors;
    size_t neighbor_count = 0;

    float seed_num = 0.0f;
    float seed_den = 0.0f;

    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            if (dx == 0 && dy == 0)
                continue;

            const int nx = xi + dx;
            const int ny = yi + dy;
            const float sample_confidence = gvl(nx, ny);
            const float sample_height = ggl(nx, ny);

            if (!std::isfinite(sample_confidence) || !std::isfinite(sample_height) || sample_confidence < min_confidence)
                continue;

            const float distance = std::hypot(static_cast<float>(dx), static_cast<float>(dy)) * resolution;
            const float spatial_weight = std::exp(-0.5f * (distance * distance) / (spatial_sigma * spatial_sigma));
            neighbors[neighbor_count++] = {dx, dy, sample_height, sample_confidence, distance, spatial_weight};

            const float base_weight = sample_confidence * spatial_weight;
            seed_num += base_weight * sample_height;
            seed_den += base_weight;
        }
    }

    if (neighbor_count > 0)
    {
        const float center_seed = seed_den > min_weight ? seed_num / seed_den : height;

        auto read_sample = [&](const int ox, const int oy, float& sample_height, float& sample_confidence) {
            sample_height = ggl(xi + ox, yi + oy);
            sample_confidence = gvl(xi + ox, yi + oy);
            return std::isfinite(sample_height) && std::isfinite(sample_confidence) && sample_confidence >= min_confidence;
        };

        float grad_x_num = 0.0f;
        float grad_x_den = 0.0f;
        for (int oy = -1; oy <= 1; ++oy)
        {
            float east_height = 0.0f, east_conf = 0.0f;
            float west_height = 0.0f, west_conf = 0.0f;
            const bool has_east = read_sample(1, oy, east_height, east_conf);
            const bool has_west = read_sample(-1, oy, west_height, west_conf);

            if (has_east && has_west)
            {
                const float pair_confidence = 0.5f * (east_conf + west_conf);
                grad_x_num += pair_confidence * (east_height - west_height) / (2.0f * resolution);
                grad_x_den += pair_confidence;
            }
            else if (has_east)
            {
                const float pair_confidence = 0.5f * east_conf;
                grad_x_num += pair_confidence * (east_height - center_seed) / resolution;
                grad_x_den += pair_confidence;
            }
            else if (has_west)
            {
                const float pair_confidence = 0.5f * west_conf;
                grad_x_num += pair_confidence * (center_seed - west_height) / resolution;
                grad_x_den += pair_confidence;
            }
        }

        float grad_y_num = 0.0f;
        float grad_y_den = 0.0f;
        for (int ox = -1; ox <= 1; ++ox)
        {
            float north_height = 0.0f, north_conf = 0.0f;
            float south_height = 0.0f, south_conf = 0.0f;
            const bool has_north = read_sample(ox, 1, north_height, north_conf);
            const bool has_south = read_sample(ox, -1, south_height, south_conf);

            if (has_north && has_south)
            {
                const float pair_confidence = 0.5f * (north_conf + south_conf);
                grad_y_num += pair_confidence * (north_height - south_height) / (2.0f * resolution);
                grad_y_den += pair_confidence;
            }
            else if (has_north)
            {
                const float pair_confidence = 0.5f * north_conf;
                grad_y_num += pair_confidence * (north_height - center_seed) / resolution;
                grad_y_den += pair_confidence;
            }
            else if (has_south)
            {
                const float pair_confidence = 0.5f * south_conf;
                grad_y_num += pair_confidence * (center_seed - south_height) / resolution;
                grad_y_den += pair_confidence;
            }
        }

        const float grad_x = grad_x_den > min_weight ? grad_x_num / grad_x_den : 0.0f;
        const float grad_y = grad_y_den > min_weight ? grad_y_num / grad_y_den : 0.0f;
        const float grad_mag = std::hypot(grad_x, grad_y);

        float normal_x = 0.0f;
        float normal_y = 0.0f;
        float tangent_x = 0.0f;
        float tangent_y = 0.0f;
        if (grad_mag > min_gradient)
        {
            normal_x = grad_x / grad_mag;
            normal_y = grad_y / grad_mag;
            tangent_x = -normal_y;
            tangent_y = normal_x;
        }

        float weighted_height_sum = 0.0f;
        float total_weight = 0.0f;

        for (size_t i = 0; i < neighbor_count; ++i)
        {
            const NeighborSample& n = neighbors[i];
            const float offset_x = static_cast<float>(n.dx) * resolution;
            const float offset_y = static_cast<float>(n.dy) * resolution;
            const float predicted_height = center_seed + grad_x * offset_x + grad_y * offset_y;
            const float height_error = std::abs(n.sample_height - predicted_height);
            const float slope_error = height_error / std::max(n.distance, 1e-4f);

            const float normalized_height_error = height_error / sigma_height;
            const float normalized_slope_error = slope_error / sigma_slope;
            const float height_weight = std::exp(-0.5f * normalized_height_error * normalized_height_error);
            const float slope_weight = std::exp(-0.5f * normalized_slope_error * normalized_slope_error);

            float directional_weight = 1.0f;
            if (grad_mag > min_gradient && n.distance > 1e-4f)
            {
                const float dir_x = offset_x / n.distance;
                const float dir_y = offset_y / n.distance;
                const float across_gradient = std::abs(dir_x * normal_x + dir_y * normal_y);
                const float along_slope = std::abs(dir_x * tangent_x + dir_y * tangent_y);
                directional_weight = std::exp(-anisotropic_factor * grad_mag * across_gradient) * (0.5f + 0.5f * along_slope);
            }

            const float weight = n.sample_confidence * n.spatial_weight * height_weight * slope_weight * directional_weight;
            weighted_height_sum += weight * n.sample_height;
            total_weight += weight;
        }

        const float interpolated_height = total_weight > min_weight ? weighted_height_sum / total_weight : center_seed;
        height = (1.0f - occupied) * interpolated_height + occupied * height;
    }

    // Only update confidence in cells above min distance
    if ((std::pow(static_cast<float>(x) - center_idx, 2.0) + std::pow(static_cast<float>(y) - center_idx, 2.0)) * std::pow(map.getResolution(), 2.0f) > minDistSquared)
        occupied = std::max(occupied - occupied / mConfig.occupied_cells_decrease_factor, 0.001);
}
