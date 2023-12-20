#include <gpd/grasp_detector.h>

namespace gpd {

GraspDetector::GraspDetector(GraspDetectionParameters& param) {
  Eigen::initParallel();

  // Read parameters from configuration file.
  param_ = param;
  outer_diameter_ = param_.hand_search_params.hand_geometry_.outer_diameter_;
  candidates_generator_ = std::make_unique<candidate::CandidatesGenerator>(
      param_.generator_params, param_.hand_search_params);
  if (!param_.model_file_.empty() || !param_.weights_file_.empty()) {
    classifier_ = net::Classifier::create(
        param_.model_file_, param_.weights_file_, static_cast<net::Classifier::Device>(param_.device_));
    // min_score_ = config_file.getValueOfKey<int>("min_score", 0);
    // printf("============ CLASSIFIER ======================\n");
    // printf("model_file: %s\n", model_file.c_str());
    // printf("weights_file: %s\n", weights_file.c_str());
    // printf("batch_size: %d\n", batch_size);
    // printf("==============================================\n");
  }

  // Create object to create grasp images from grasp candidates (used for
  // classification).
  image_generator_ = std::make_unique<descriptor::ImageGenerator>(
      param_.image_params, param_.hand_search_params.num_threads_,
      param_.hand_search_params.num_orientations_, false, param_.plot_params.remove_plane_);

  min_aperture_ = param_.gripper_width_range_[0];
  max_aperture_ = param_.gripper_width_range_[1];

  // Read clustering parameters.
  clustering_ = std::make_unique<Clustering>(param_.min_inliers_);
  cluster_grasps_ = param_.min_inliers_ > 0 ? true : false;

  // Create plotter.
  plotter_ = std::make_unique<util::Plot>(param_.hand_search_params.hand_axes_.size(),
                                          param_.hand_search_params.num_orientations_);
}

std::vector<std::unique_ptr<candidate::Hand>> GraspDetector::detectGrasps(
    const util::Cloud &cloud) {
  double t0_total = omp_get_wtime();
  std::vector<std::unique_ptr<candidate::Hand>> hands_out;

  const candidate::HandGeometry &hand_geom =
      candidates_generator_->getHandSearchParams().hand_geometry_;

  // Check if the point cloud is empty.
  if (cloud.getCloudOriginal()->size() == 0) {
    printf("ERROR: Point cloud is empty!");
    hands_out.resize(0);
    return hands_out;
  }

  // Plot samples/indices.
  if (param_.plot_params.plot_samples_) {
    if (cloud.getSamples().cols() > 0) {
      plotter_->plotSamples(cloud.getSamples(), cloud.getCloudProcessed());
    } else if (cloud.getSampleIndices().size() > 0) {
      plotter_->plotSamples(cloud.getSampleIndices(),
                            cloud.getCloudProcessed());
    }
  }

  if (param_.plot_params.plot_normals_) {
    std::cout << "Plotting normals for different camera sources\n";
    plotter_->plotNormals(cloud);
  }

  // 1. Generate grasp candidates.
  double t0_candidates = omp_get_wtime();
  std::vector<std::unique_ptr<candidate::HandSet>> hand_set_list =
      candidates_generator_->generateGraspCandidateSets(cloud);
  printf("Generated %zu hand sets.\n", hand_set_list.size());
  if (hand_set_list.size() == 0) {
    return hands_out;
  }
  double t_candidates = omp_get_wtime() - t0_candidates;
  if (param_.plot_params.plot_candidates_) {
    plotter_->plotFingers3D(hand_set_list, cloud.getCloudOriginal(),
                            "Grasp candidates", hand_geom);
  }

  // 2. Filter the candidates.
  double t0_filter = omp_get_wtime();
  std::vector<std::unique_ptr<candidate::HandSet>> hand_set_list_filtered =
      filterGraspsWorkspace(hand_set_list, param_.workspace_grasps_);
  if (hand_set_list_filtered.size() == 0) {
    return hands_out;
  }
  if (param_.plot_params.plot_filtered_candidates_) {
    plotter_->plotFingers3D(hand_set_list_filtered, cloud.getCloudOriginal(),
                            "Filtered Grasps (Aperture, Workspace)", hand_geom);
  }
  if (param_.filter_approach_direction_) {
    hand_set_list_filtered =
        filterGraspsDirection(hand_set_list_filtered, direction_, thresh_rad_);
    if (param_.plot_params.plot_filtered_candidates_) {
      plotter_->plotFingers3D(hand_set_list_filtered, cloud.getCloudOriginal(),
                              "Filtered Grasps (Approach)", hand_geom);
    }
  }
  double t_filter = omp_get_wtime() - t0_filter;
  if (hand_set_list_filtered.size() == 0) {
    return hands_out;
  }

  // 3. Create grasp descriptors (images).
  double t0_images = omp_get_wtime();
  std::vector<std::unique_ptr<candidate::Hand>> hands;
  std::vector<std::unique_ptr<cv::Mat>> images;
  image_generator_->createImages(cloud, hand_set_list_filtered, images, hands);
  double t_images = omp_get_wtime() - t0_images;

  // 4. Classify the grasp candidates.
  double t0_classify = omp_get_wtime();
  std::vector<float> scores = classifier_->classifyImages(images);
  for (int i = 0; i < hands.size(); i++) {
    hands[i]->setScore(scores[i]);
  }
  double t_classify = omp_get_wtime() - t0_classify;

  // 5. Select the <num_selected> highest scoring grasps.
  hands = selectGrasps(hands);
  if (param_.plot_params.plot_valid_grasps_) {
    plotter_->plotFingers3D(hands, cloud.getCloudOriginal(), "Valid Grasps",
                            hand_geom);
  }

  // 6. Cluster the grasps.
  double t0_cluster = omp_get_wtime();
  std::vector<std::unique_ptr<candidate::Hand>> clusters;
  if (cluster_grasps_) {
    clusters = clustering_->findClusters(hands);
    printf("Found %d clusters.\n", (int)clusters.size());
    if (clusters.size() <= 3) {
      printf(
          "Not enough clusters found! Adding all grasps from previous step.");
      for (int i = 0; i < hands.size(); i++) {
        clusters.push_back(std::move(hands[i]));
      }
    }
    if (param_.plot_params.plot_clustered_grasps_) {
      plotter_->plotFingers3D(clusters, cloud.getCloudOriginal(),
                              "Clustered Grasps", hand_geom);
    }
  } else {
    clusters = std::move(hands);
  }
  double t_cluster = omp_get_wtime() - t0_cluster;

  // 7. Sort grasps by their score.
  std::sort(clusters.begin(), clusters.end(), isScoreGreater);
  printf("======== Selected grasps ========\n");
  for (int i = 0; i < clusters.size(); i++) {
    std::cout << "Grasp " << i << ": " << clusters[i]->getScore() << "\n";
  }
  printf("Selected the %d best grasps.\n", (int)clusters.size());
  double t_total = omp_get_wtime() - t0_total;

  printf("======== RUNTIMES ========\n");
  printf(" 1. Candidate generation: %3.4fs\n", t_candidates);
  printf(" 2. Descriptor extraction: %3.4fs\n", t_images);
  printf(" 3. Classification: %3.4fs\n", t_classify);
  // printf(" Filtering: %3.4fs\n", t_filter);
  // printf(" Clustering: %3.4fs\n", t_cluster);
  printf("==========\n");
  printf(" TOTAL: %3.4fs\n", t_total);

  if (param_.plot_params.plot_selected_grasps_) {
    plotter_->plotFingers3D(clusters, cloud.getCloudOriginal(),
                            "Selected Grasps", hand_geom, false);
  }

  return clusters;
}

void GraspDetector::preprocessPointCloud(util::Cloud &cloud) {
  candidates_generator_->preprocessPointCloud(cloud);
}

std::vector<std::unique_ptr<candidate::HandSet>>
GraspDetector::filterGraspsWorkspace(
    std::vector<std::unique_ptr<candidate::HandSet>> &hand_set_list,
    const std::vector<double> &workspace) const {
  int remaining = 0;
  std::vector<std::unique_ptr<candidate::HandSet>> hand_set_list_out;
  printf("Filtering grasps outside of workspace ...\n");

  const candidate::HandGeometry &hand_geometry =
      candidates_generator_->getHandSearchParams().hand_geometry_;

  for (int i = 0; i < hand_set_list.size(); i++) {
    const std::vector<std::unique_ptr<candidate::Hand>> &hands =
        hand_set_list[i]->getHands();
    Eigen::Array<bool, 1, Eigen::Dynamic> is_valid =
        hand_set_list[i]->getIsValid();

    for (int j = 0; j < hands.size(); j++) {
      if (!is_valid(j)) {
        continue;
      }
      double half_width = 0.5 * hand_geometry.outer_diameter_;
      Eigen::Vector3d left_bottom =
          hands[j]->getPosition() + half_width * hands[j]->getBinormal();
      Eigen::Vector3d right_bottom =
          hands[j]->getPosition() - half_width * hands[j]->getBinormal();
      Eigen::Vector3d left_top =
          left_bottom + hand_geometry.depth_ * hands[j]->getApproach();
      Eigen::Vector3d right_top =
          left_bottom + hand_geometry.depth_ * hands[j]->getApproach();
      Eigen::Vector3d approach =
          hands[j]->getPosition() - 0.05 * hands[j]->getApproach();
      Eigen::VectorXd x(5), y(5), z(5);
      x << left_bottom(0), right_bottom(0), left_top(0), right_top(0),
          approach(0);
      y << left_bottom(1), right_bottom(1), left_top(1), right_top(1),
          approach(1);
      z << left_bottom(2), right_bottom(2), left_top(2), right_top(2),
          approach(2);

      // Ensure the object fits into the hand and avoid grasps outside the
      // workspace.
      if (hands[j]->getGraspWidth() >= min_aperture_ &&
          hands[j]->getGraspWidth() <= max_aperture_ &&
          x.minCoeff() >= workspace[0] && x.maxCoeff() <= workspace[1] &&
          y.minCoeff() >= workspace[2] && y.maxCoeff() <= workspace[3] &&
          z.minCoeff() >= workspace[4] && z.maxCoeff() <= workspace[5]) {
        is_valid(j) = true;
        remaining++;
      } else {
        is_valid(j) = false;
      }
    }

    if (is_valid.any()) {
      hand_set_list_out.push_back(std::move(hand_set_list[i]));
      hand_set_list_out[hand_set_list_out.size() - 1]->setIsValid(is_valid);
    }
  }

  printf("Number of grasp candidates within workspace and gripper width: %d\n",
         remaining);

  return hand_set_list_out;
}

std::vector<std::unique_ptr<candidate::HandSet>>
GraspDetector::generateGraspCandidates(const util::Cloud &cloud) {
  return candidates_generator_->generateGraspCandidateSets(cloud);
}

std::vector<std::unique_ptr<candidate::Hand>> GraspDetector::selectGrasps(
    std::vector<std::unique_ptr<candidate::Hand>> &hands) const {
  printf("Selecting the %d highest scoring grasps ...\n", param_.num_selected_);

  int middle = std::min((int)hands.size(), param_.num_selected_);
  std::partial_sort(hands.begin(), hands.begin() + middle, hands.end(),
                    isScoreGreater);
  std::vector<std::unique_ptr<candidate::Hand>> hands_out;

  for (int i = 0; i < middle; i++) {
    hands_out.push_back(std::move(hands[i]));
    printf(" grasp #%d, score: %3.4f\n", i, hands_out[i]->getScore());
  }

  return hands_out;
}

std::vector<std::unique_ptr<candidate::HandSet>>
GraspDetector::filterGraspsDirection(
    std::vector<std::unique_ptr<candidate::HandSet>> &hand_set_list,
    const Eigen::Vector3d &direction, const double thresh_rad) {
  std::vector<std::unique_ptr<candidate::HandSet>> hand_set_list_out;
  int remaining = 0;

  for (int i = 0; i < hand_set_list.size(); i++) {
    const std::vector<std::unique_ptr<candidate::Hand>> &hands =
        hand_set_list[i]->getHands();
    Eigen::Array<bool, 1, Eigen::Dynamic> is_valid =
        hand_set_list[i]->getIsValid();

    for (int j = 0; j < hands.size(); j++) {
      if (is_valid(j)) {
        double angle = acos(direction.transpose() * hands[j]->getApproach());
        if (angle > thresh_rad) {
          is_valid(j) = false;
        } else {
          remaining++;
        }
      }
    }

    if (is_valid.any()) {
      hand_set_list_out.push_back(std::move(hand_set_list[i]));
      hand_set_list_out[hand_set_list_out.size() - 1]->setIsValid(is_valid);
    }
  }

  printf("Number of grasp candidates with correct approach direction: %d\n",
         remaining);

  return hand_set_list_out;
}

bool GraspDetector::createGraspImages(
    util::Cloud &cloud,
    std::vector<std::unique_ptr<candidate::Hand>> &hands_out,
    std::vector<std::unique_ptr<cv::Mat>> &images_out) {
  // Check if the point cloud is empty.
  if (cloud.getCloudOriginal()->size() == 0) {
    printf("ERROR: Point cloud is empty!");
    hands_out.resize(0);
    images_out.resize(0);
    return false;
  }

  // Plot samples/indices.
  if (param_.plot_params.plot_samples_) {
    if (cloud.getSamples().cols() > 0) {
      plotter_->plotSamples(cloud.getSamples(), cloud.getCloudProcessed());
    } else if (cloud.getSampleIndices().size() > 0) {
      plotter_->plotSamples(cloud.getSampleIndices(),
                            cloud.getCloudProcessed());
    }
  }

  if (param_.plot_params.plot_normals_) {
    std::cout << "Plotting normals for different camera sources\n";
    plotter_->plotNormals(cloud);
  }

  // 1. Generate grasp candidates.
  std::vector<std::unique_ptr<candidate::HandSet>> hand_set_list =
      candidates_generator_->generateGraspCandidateSets(cloud);
  printf("Generated %zu hand sets.\n", hand_set_list.size());
  if (hand_set_list.size() == 0) {
    hands_out.resize(0);
    images_out.resize(0);
    return false;
  }

  const candidate::HandGeometry &hand_geom =
      candidates_generator_->getHandSearchParams().hand_geometry_;

  // 2. Filter the candidates.
  std::vector<std::unique_ptr<candidate::HandSet>> hand_set_list_filtered =
      filterGraspsWorkspace(hand_set_list, param_.workspace_grasps_);
  if (param_.plot_params.plot_filtered_candidates_) {
    plotter_->plotFingers3D(hand_set_list_filtered, cloud.getCloudOriginal(),
                            "Filtered Grasps (Aperture, Workspace)", hand_geom);
  }
  if (filter_approach_direction_) {
    hand_set_list_filtered =
        filterGraspsDirection(hand_set_list_filtered, direction_, thresh_rad_);
    if (param_.plot_params.plot_filtered_candidates_) {
      plotter_->plotFingers3D(hand_set_list_filtered, cloud.getCloudOriginal(),
                              "Filtered Grasps (Approach)", hand_geom);
    }
  }

  // 3. Create grasp descriptors (images).
  std::vector<std::unique_ptr<candidate::Hand>> hands;
  std::vector<std::unique_ptr<cv::Mat>> images;
  image_generator_->createImages(cloud, hand_set_list_filtered, images_out,
                                 hands_out);

  return true;
}

std::vector<int> GraspDetector::evalGroundTruth(
    const util::Cloud &cloud_gt,
    std::vector<std::unique_ptr<candidate::Hand>> &hands) {
  return candidates_generator_->reevaluateHypotheses(cloud_gt, hands);
}

std::vector<std::unique_ptr<candidate::Hand>>
GraspDetector::pruneGraspCandidates(
    const util::Cloud &cloud,
    const std::vector<std::unique_ptr<candidate::HandSet>> &hand_set_list,
    double min_score) {
  // 1. Create grasp descriptors (images).
  std::vector<std::unique_ptr<candidate::Hand>> hands;
  std::vector<std::unique_ptr<cv::Mat>> images;
  image_generator_->createImages(cloud, hand_set_list, images, hands);

  // 2. Classify the grasp candidates.
  std::vector<float> scores = classifier_->classifyImages(images);
  std::vector<std::unique_ptr<candidate::Hand>> hands_out;

  // 3. Only keep grasps with a score larger than <min_score>.
  for (int i = 0; i < hands.size(); i++) {
    if (scores[i] > min_score) {
      hands[i]->setScore(scores[i]);
      hands_out.push_back(std::move(hands[i]));
    }
  }

  return hands_out;
}

void GraspDetector::printStdVector(const std::vector<int> &v,
                                   const std::string &name) const {
  printf("%s: ", name.c_str());
  for (int i = 0; i < v.size(); i++) {
    printf("%d ", v[i]);
  }
  printf("\n");
}

void GraspDetector::printStdVector(const std::vector<double> &v,
                                   const std::string &name) const {
  printf("%s: ", name.c_str());
  for (int i = 0; i < v.size(); i++) {
    printf("%3.2f ", v[i]);
  }
  printf("\n");
}

}  // namespace gpd
