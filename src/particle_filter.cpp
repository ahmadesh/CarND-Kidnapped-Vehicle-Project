/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

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
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    
    num_particles = 1000;
    
    default_random_engine gen;
    
    // This line creates a normal (Gaussian) distribution for x
    normal_distribution<double> dist_x(x, std[0]);
    
    // Create normal distributions for y and theta
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[3]);
    
    for (int i = 0; i < num_particles; ++i) {
        Particle p;
        
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1;
        p.id = i;
        
        particles.push_back(p);
        weights.push_back(1.0);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    double x;
    double y;
    double theta;
    default_random_engine gen;
    
    for (int i = 0; i < num_particles; ++i) {
        Particle p = particles[i];
        
        if (abs(yaw_rate)<0.0001) {
            x = p.x + velocity * delta_t * cos(p.theta);
            y = p.y + velocity * delta_t * sin(p.theta);
            theta = yaw_rate * delta_t;
        }
        else {
            x = p.x + velocity/yaw_rate * (sin(p.theta+yaw_rate*delta_t) - sin(p.theta));
            y = p.y + velocity/yaw_rate * (-cos(p.theta+yaw_rate*delta_t) + cos(p.theta));
            theta = p.theta + yaw_rate*delta_t;
        }
        
        normal_distribution<double> dist_x(x, std_pos[0]);
        normal_distribution<double> dist_y(y, std_pos[1]);
        normal_distribution<double> dist_theta(theta, std_pos[3]);
        
        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    
    
    for (unsigned int i=0; i<observations.size();i++){
        double mindist = numeric_limits<double>::max();
        int id = -1;
        for (auto pred : predicted) {
            double dist = sqrt(pow(pred.x-observations[i].x,2) + pow(pred.y-observations[i].y,2));
            if (dist<mindist) {
                mindist=dist;
                id = pred.id;
            }
        }
        observations[i].id = id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    
    for (unsigned int i=0; i < num_particles; i++) {
        // transform to map x coordinate
        double x = particles[i].x;
        double y = particles[i].y;
        double theta = particles[i].theta;
        
        vector<LandmarkObs> predictions;
        
        for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
            
            // get id and x,y coordinates
            float lm_x = map_landmarks.landmark_list[j].x_f;
            float lm_y = map_landmarks.landmark_list[j].y_f;
            int lm_id = map_landmarks.landmark_list[j].id_i;
            
            // only consider landmarks within sensor range
            if (fabs(lm_x - x) <= sensor_range && fabs(lm_y - y) <= sensor_range) {
                // add prediction to vector
                predictions.push_back(LandmarkObs{lm_id, lm_x, lm_y});
            }
         }
    
        vector<LandmarkObs> obesrvations_tf;
        for (unsigned int k=0; k < observations.size(); k++) {
            double x_map = x + (cos(theta) * observations[k].x) - (sin(theta) * observations[k].y);
            double y_map = y + (sin(theta) * observations[k].x) + (cos(theta) * observations[k].y);
            obesrvations_tf.push_back(LandmarkObs{-1, x_map, y_map});
        }
        
        dataAssociation(predictions, obesrvations_tf);
        
        weights[i] = 1;
        double gauss_norm= 1.0/(2.0 * M_PI * std_landmark[0] * std_landmark[1]);
        for (unsigned int k=0; k < obesrvations_tf.size(); k++) {
            for (unsigned int j=0; j < predictions.size(); j++) {
                if (predictions[j].id == obesrvations_tf[k].id) {
                    double x_obs = obesrvations_tf[k].x;
                    double y_obs = obesrvations_tf[k].y;
                    double mu_x = predictions[j].x;
                    double mu_y = predictions[j].y;
                    
                    double exponent = pow(x_obs - mu_x, 2)/(2 * pow(std_landmark[0],2)) + pow(y_obs - mu_y,2)/(2 * pow(std_landmark[1], 2));
                    weights[i] = weights[i]  * gauss_norm * exp(-exponent);
                }
            }
        }
        particles[i].weight = weights[i];
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> distribution(weights.begin(), weights.end());
    
    std::vector<Particle> new_particles;
    for (unsigned int i=0; i<num_particles; i++) {
        new_particles.push_back(particles[distribution(gen)]);
    }
    particles = new_particles;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
