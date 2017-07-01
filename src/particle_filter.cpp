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
#include <cfloat>
#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::VectorXd;

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 2;
	auto sigma_x = std[0];
	auto sigma_y = std[1];
	auto sigma_theta = std[2];
	default_random_engine gen;
	normal_distribution<double> dist_x(x, sigma_x);
	normal_distribution<double> dist_y(y, sigma_y);
	normal_distribution<double> dist_theta(theta, sigma_theta);
	for (auto count = 0; count < num_particles; ++count) {
		double new_x = dist_x(gen);
		double new_y = dist_y(gen);
		double new_theta = dist_theta(gen);
		Particle newParticle;
		newParticle.id = count;
		newParticle.x = new_x;
		newParticle.y = new_y;
		newParticle.x = new_theta;
		newParticle.weight = .0;
		particles.push_back(newParticle);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
		double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	auto sigma_x = std_pos[0];
	auto sigma_y = std_pos[1];
	auto sigma_theta = std_pos[2];
	default_random_engine gen;
	normal_distribution<double> dist_x(0, sigma_x);
	normal_distribution<double> dist_y(0, sigma_y);
	normal_distribution<double> dist_theta(0, sigma_theta);

	for (auto & particle: particles) {
		double xUpdated, yUpdated, thetaUpdated;
		if (yaw_rate== 0) {
			xUpdated=particle.x+velocity*delta_t*cos(particle.theta);
			yUpdated=particle.y+velocity*delta_t*sin(particle.theta);
			thetaUpdated=particle.theta;
		}
		else {
			xUpdated=particle.x+velocity/yaw_rate*(sin(particle.theta+yaw_rate*delta_t)-sin(particle.theta));
			yUpdated=particle.y+velocity/yaw_rate*(cos(particle.theta)-cos(particle.theta+yaw_rate*delta_t));
			thetaUpdated=particle.theta+yaw_rate*delta_t;
		}
		particle.x= xUpdated+dist_x(gen);
		particle.y=yUpdated+dist_y(gen);
		particle.theta=thetaUpdated+dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
		std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
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

	auto maxSqDist = pow(sensor_range,2);
	auto covariance=MatrixXd(2,2);
	covariance << std_landmark[0], 0, 0, std_landmark[1];
	MatrixXd invCovariance = covariance.inverse();

	double weightsSum=0;
	for(auto & particle: particles) {
		// Recompute the observations in the particle reference system, and write them in observationsGlobalRef
		vector<LandmarkObs> observationsGlobalRef;
		for (auto observation: observations) {
			auto theta = -particle.theta;
			auto x_t= -particle.x;
			auto y_t = -particle.y;
			LandmarkObs landmarkGlobalRef;
			landmarkGlobalRef.x = observation.x*cos(theta)-observation.y*sin(theta)+x_t;
			landmarkGlobalRef.y = observation.x*sin(theta)+observation.y*cos(theta)+y_t;
			landmarkGlobalRef.id = -1;  // Flag it hasn't been set yet
			observationsGlobalRef.push_back(landmarkGlobalRef);
		}

		// Set the .id of every item (observation) in observationsGlobalRef to the id of the closest landmark
		for (auto & observation: observationsGlobalRef) {
			// Find the landmark which is the closest to the observation
			double bestSqDist=DBL_MAX;
			for (auto landmark: map_landmarks.landmark_list){
				// Compute the square of the landmark-observation distance
				auto sqDist= pow((observation.x-landmark.x_f),2)+pow((observation.y-landmark.y_f),2);
				// Keep track of the closest match so far
				if (sqDist < bestSqDist && sqDist<= maxSqDist) {
					bestSqDist=sqDist;
					observation.id=landmark.id_i;
				}
			}
			assert(observation.id!=-1);  // If no measurement of landmark if within sensor range, we are in troubles!
		}

		// Compute the weight for the given particle
		double weight=1;
		for (auto observation: observationsGlobalRef) {
			VectorXd x(2);
			x<< observation.x, observation.y;
			VectorXd mu(2);
			mu << map_landmarks.landmark_list[observation.id].x_f, map_landmarks.landmark_list[observation.id].y_f ;
			VectorXd xDiff =x-mu;
			double res= xDiff.transpose()*invCovariance*xDiff;
			weight*=exp(-res/2);
		}
		particle.weight = weight;
		weightsSum+=weight;
	}

	// Now all particles have the weights updated, but they still need to be normalised (ensure they add up to 1)
	for (auto & particle: particles) {
		particle.weight/=weightsSum;
	}



}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

Particle ParticleFilter::SetAssociations(Particle particle,
		std::vector<int> associations, std::vector<double> sense_x,
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
