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
#include <cassert>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::initialise(double x, double y, double theta, double std[], unsigned numParticlesInit) {
	numParticles = numParticlesInit;

	/*
	 * Initialise the random number generators
	 */
	auto sigma_x = std[0];
	auto sigma_y = std[1];
	auto sigma_theta = std[2];
	normal_distribution<double> dist_x(x, sigma_x);
	normal_distribution<double> dist_y(y, sigma_y);
	normal_distribution<double> dist_theta(theta, sigma_theta);

	/*
	 * Draw the initial filter particles
	 */
	for (auto count = 0; count < numParticles; ++count) {
		Particle newParticle;
		newParticle.id = count;
		newParticle.x = dist_x(randomGenerator);
		newParticle.y = dist_y(randomGenerator);
		newParticle.theta = dist_theta(randomGenerator);
		newParticle.weight = 1.;
		particles.push_back(newParticle);
	}

	// Initialisation done
	isInitialized = true;
}

void ParticleFilter::predict(double delta_t, double std_pos[],
		double velocity, double yaw_rate) {
	/*
	 * Initialise the random number generators
	 */
	auto sigma_x = std_pos[0];
	auto sigma_y = std_pos[1];
	auto sigma_theta = std_pos[2];
	normal_distribution<double> dist_x(0, sigma_x);
	normal_distribution<double> dist_y(0, sigma_y);
	normal_distribution<double> dist_theta(0, sigma_theta);

	/*
	 * Update every particle position based on the process model
	 */
	for (auto & particle: particles) {
		if (yaw_rate== 0) {
			particle.x+=velocity*delta_t*cos(particle.theta);
			particle.y+=velocity*delta_t*sin(particle.theta);
			// Note: particle.theta remains unchanged
		}
		else {
			particle.x+=velocity/yaw_rate*(sin(particle.theta+yaw_rate*delta_t)-sin(particle.theta));
			particle.y+=velocity/yaw_rate*(cos(particle.theta)-cos(particle.theta+yaw_rate*delta_t));
			particle.theta+=yaw_rate*delta_t;
		}

		/*
		 * Randomly perturb every particle pose, after update, to keep the process noise into account.
		 * If you don't do it, the particle filter accuracy will suffer.
		 */
		particle.x+=dist_x(randomGenerator);
		particle.y+=dist_y(randomGenerator);
		particle.theta+=dist_theta(randomGenerator);
	}
}

vector<LandmarkObs> ParticleFilter::convertLocalToGlobal(const double x, const double y, const double theta, vector<LandmarkObs> observations) {
	vector<LandmarkObs> observationsGlobalRef;  // Return value
	for (auto observation: observations) {
		LandmarkObs landmarkGlobalRef;
		landmarkGlobalRef.x = observation.x*cos(theta)-observation.y*sin(theta)+x;
		landmarkGlobalRef.y = observation.x*sin(theta)+observation.y*cos(theta)+y;
		landmarkGlobalRef.id = -1;  // Flag it hasn't been set yet
		observationsGlobalRef.push_back(landmarkGlobalRef);
	}
	return observationsGlobalRef;
}


void ParticleFilter::matchObservationsWithLandmarks(vector<LandmarkObs> & observationsGlobalRef, const vector<Map::single_landmark_s> landmarks) {
	// Set the .id of every item (observation) in observationsGlobalRef to the index of the closest landmark in vector parameter landmarks
	for (auto & observation: observationsGlobalRef) {
		// Find the landmark which is the closest to the observation
		double bestSqDist=DBL_MAX;  // Square distance of the closes landmark so far
		for (unsigned iLandmark=0; iLandmark< landmarks.size(); ++iLandmark){
			// Compute the square of the landmark-observation distance
			auto sqDist= pow((observation.x-landmarks[iLandmark].x_f),2)+pow((observation.y-landmarks[iLandmark].y_f),2);
			// Keep track of the closest match so far
			if (sqDist < bestSqDist) {
				bestSqDist=sqDist;
				observation.id=iLandmark;  // Note, observation.id is set to the index position of the landmark in vector `landmarks`;
				   // it is *not* set to landmarks[iLandmark].id_i
			}
		}
	}
}


void ParticleFilter::updateWeights(double sensorRange, double std_landmark[],
		std::vector<LandmarkObs> observations, Map mapLandmarks) {
	for(auto & particle: particles) {
		// Recompute the observations in the particle reference system, and write them in observationsGlobalRef
		vector<LandmarkObs> observationsGlobalRef=convertLocalToGlobal(particle.x, particle.y, particle.theta, observations);

		/*
		 * Collect all the landmarks that are within sensor range
		 */
		vector<Map::single_landmark_s> landmarksInRange;
		for (auto landmark : mapLandmarks.landmark_list)
			if (sqrt(pow(landmark.x_f-particle.x,2)+pow(landmark.y_f-particle.y, 2))<= sensorRange)
				landmarksInRange.push_back(landmark);

		// Associate every observation with the observed landmark
		matchObservationsWithLandmarks(observationsGlobalRef, landmarksInRange);

		/*
		 *  Compute the weight for the given particle
		 */
		double weight=1;
		// Empty visualisation and debugging information, to fill it in below
		particle.associations.clear();
		particle.sense_x.clear();
		particle.sense_y.clear();
		// Use observations to compute the weight
		for (auto observation: observationsGlobalRef) {
			auto mu_x= landmarksInRange[observation.id].x_f;
			auto mu_y= landmarksInRange[observation.id].y_f;
			auto sigma_x = std_landmark[0];
			auto sigma_y = std_landmark[1];
			weight*= exp(-(pow(observation.x-mu_x,2)/(2*pow(sigma_x,2))+pow(observation.y-mu_y,2)/(2*pow(sigma_y,2))))/(2*M_PI*sigma_x*sigma_y);
			// Fill-in visualisation and debugging information
			particle.associations.push_back(landmarksInRange[observation.id].id_i);
			particle.sense_x.push_back(observation.x);
			particle.sense_y.push_back(observation.y);
		}
		particle.weight = weight;
	}

}

void ParticleFilter::resample() {
	vector<Particle> resampled;
	vector<double> weights;
	for (auto particle: particles)
		weights.push_back(particle.weight);
	discrete_distribution<> dist(begin(weights), end(weights));
	for (auto count=0; count< numParticles; ++count) {
		/* Draw a random number between 0 and the number of particles minus 1, from the discrete
		 * distribution given by weights. Note that elements in `weights` doesn't need to add up to 1,
		 * discrete_distribution<> works correctly anyway.
		 */
		auto randomIndex= dist(randomGenerator);
		assert(randomIndex>=0);
		assert(static_cast<unsigned>(randomIndex)<particles.size());
		resampled.push_back(particles[randomIndex]);
	}
	particles=resampled;
}

Particle ParticleFilter::getBestParticle() const {
	double highest_weight = -1.0;
	Particle bestParticle;
	for (const auto particle: particles)
		if (particle.weight > highest_weight) {
			highest_weight = particle.weight;
			bestParticle = particle;
		}
	return bestParticle;
}

// Code below coming from Udacity's starter code

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
