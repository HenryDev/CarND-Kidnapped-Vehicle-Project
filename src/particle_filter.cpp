#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 100;
    default_random_engine random_engine;
    normal_distribution<double> n_x(x, std[0]);
    normal_distribution<double> n_y(y, std[1]);
    normal_distribution<double> n_theta(theta, std[2]);
    for (int i = 0; i < num_particles; i++) {
        Particle particle;
        particle.id = i;
        particle.x = n_x(random_engine);
        particle.y = n_y(random_engine);
        particle.theta = n_theta(random_engine);
        particle.weight = 1;

        particles.push_back(particle);
        weights.push_back(1);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/
    default_random_engine random_engine;
    for (int i = 0; i < num_particles; i++) {
        double new_x;
        double new_y;
        double new_theta;
        Particle &particle = particles[i];
        if (yaw_rate == 0) {
            new_x = particle.x + velocity * delta_t * cos(particle.theta);
            new_y = particle.y + velocity * delta_t * sin(particle.theta);
            new_theta = particle.theta;
        } else {
            new_x = particle.x + velocity / yaw_rate * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
            new_y = particle.y + velocity / yaw_rate * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
            new_theta = particle.theta + yaw_rate * delta_t;
        }
        normal_distribution<double> n_x(new_x, std_pos[0]);
        normal_distribution<double> n_y(new_y, std_pos[1]);
        normal_distribution<double> n_theta(new_theta, std_pos[2]);

        particle.x = n_x(random_engine);
        particle.y = n_y(random_engine);
        particle.theta = n_theta(random_engine);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
    // Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.
    for (int o = 0; o < observations.size(); o++) {
        double minimum_distance = numeric_limits<double>::max();
        LandmarkObs &observation = observations[o];
        for (int p = 0; p < predicted.size(); p++) {
            LandmarkObs &prediction = predicted[p];
            double distance = dist(observation.x, observation.y, prediction.x, prediction.y);
            if (distance < minimum_distance) {
                minimum_distance = distance;
                observation.id = prediction.id;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
    // Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation 3.33
    //   http://planning.cs.uiuc.edu/node99.html

    for (int p = 0; p < num_particles; p++) {
        Particle particle = particles[p];
        vector<LandmarkObs> landmark_filtered;
        for (int l = 0; l < map_landmarks.landmark_list.size(); l++) {
            Map::single_landmark_s landmark = map_landmarks.landmark_list[l];
            double distance = dist(particle.x, particle.y, landmark.x_f, landmark.y_f);
            if (distance < sensor_range) {
                LandmarkObs nearby_landmark;
                nearby_landmark.id = landmark.id_i;
                nearby_landmark.x = landmark.x_f;
                nearby_landmark.y = landmark.y_f;
                landmark_filtered.push_back(nearby_landmark);
            }
        }

        vector<LandmarkObs> transformed_observations;
        for (int o = 0; o < observations.size(); o++) {
            LandmarkObs observation = observations[o];
            LandmarkObs transformed_observation;
            transformed_observation.x =
                    observation.x * cos(particle.theta) - observation.y * sin(particle.theta) + particle.x;
            transformed_observation.y =
                    observation.x * sin(particle.theta) + observation.y * cos(particle.theta) + particle.y;
            transformed_observations.push_back(transformed_observation);
        }

        dataAssociation(landmark_filtered, transformed_observations);

        vector<int> associations;
        vector<double> sense_x;
        vector<double> sense_y;
        for (int j = 0; j < transformed_observations.size(); j++) {
            LandmarkObs obs = transformed_observations[j];
            associations.push_back(obs.id);
            sense_x.push_back(obs.x);
            sense_y.push_back(obs.y);
        }
        particle = SetAssociations(particle, associations, sense_x, sense_y);
        particles[p] = particle;

        double weight = 1;
        for (int a = 0; a < particle.associations.size(); a++) {
            double x = particle.sense_x[a];
            double y = particle.sense_y[a];
            double predicted_x = 0;
            double predicted_y = 0;
            for (int l = 0; l < landmark_filtered.size(); l++) {
                if (landmark_filtered[l].id == particle.associations[a]) {
                    predicted_x = landmark_filtered[l].x;
                    predicted_y = landmark_filtered[l].y;
                    break;
                }
            }
            weight *= exp(-0.5 * (pow(x - predicted_x, 2) / pow(std_landmark[0], 2) +
                                  pow(y - predicted_y, 2) / pow(std_landmark[1], 2)));
        }
        particles[p].weight = weight;
        weights[p] = weight;
    }
}

void ParticleFilter::resample() {
    // Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    // http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    default_random_engine random_engine;
    discrete_distribution<int> distribution(weights.begin(), weights.end());
    vector<Particle> resample_particles;
    for (int i = 0; i < num_particles; i++) {
        resample_particles.push_back(particles[distribution(random_engine)]);
    }
    particles = resample_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best) {
    vector<double> v = best.sense_x;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best) {
    vector<double> v = best.sense_y;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}
