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

double normaliseAngle(const double angle) {
	return angle;
	//return atan2(sin(angle), cos(angle));
}


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;
	auto sigma_x = std[0];
	auto sigma_y = std[1];
	auto sigma_theta = std[2];
	default_random_engine gen;
	normal_distribution<double> dist_x(x, sigma_x);
	normal_distribution<double> dist_y(y, sigma_y);
	normal_distribution<double> dist_theta(theta, sigma_theta);
	for (auto count = 0; count < num_particles; ++count) {
		Particle newParticle;
		newParticle.id = count;
		newParticle.x = dist_x(gen);
		newParticle.y = dist_y(gen);
		newParticle.theta = dist_theta(gen);
		newParticle.theta = normaliseAngle(newParticle.theta);
		newParticle.weight = 1./num_particles;
		particles.push_back(newParticle);
	}
	is_initialized = true;
}

void ParticleFilter::testInit() {
	num_particles = 2;
	Particle newParticle1;
	newParticle1.id=1;
	newParticle1.x=10;
	newParticle1.y=20;
	newParticle1.theta=0;
	newParticle1.weight=1./num_particles;
	Particle newParticle2;
	newParticle2.id=2;
	newParticle2.x=20;
	newParticle2.y=10;
	newParticle2.weight=1./num_particles;
	newParticle2.theta=3.1415926535/2;
	particles.push_back(newParticle1);
	particles.push_back(newParticle2);
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
		if (yaw_rate== 0) {
			particle.x+=velocity*delta_t*cos(particle.theta);
			particle.y+=velocity*delta_t*sin(particle.theta);
			// particle.theta remains unchanged
		}
		else {
			particle.x+=velocity/yaw_rate*(sin(particle.theta+yaw_rate*delta_t)-sin(particle.theta));
			particle.y+=velocity/yaw_rate*(cos(particle.theta)-cos(particle.theta+yaw_rate*delta_t));
			particle.theta+=yaw_rate*delta_t;
			particle.theta= normaliseAngle(particle.theta);
		}
		if (sigma_x>0)
			particle.x+=dist_x(gen);
		if (sigma_y>0)
			particle.y+=dist_y(gen);
		if (sigma_theta>0) {
			particle.theta+=dist_theta(gen);
			particle.theta= normaliseAngle(particle.theta);
		}
	}
}

vector<LandmarkObs> ParticleFilter::convertLocalToGlobal(const double x, const double y, const double theta, vector<LandmarkObs> observations) {
	vector<LandmarkObs> observationsGlobalRef;
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
	// Set the .id of every item (observation) in observationsGlobalRef to the id of the closest landmark within sensors range
	for (auto & observation: observationsGlobalRef) {
		// Find the landmark which is the closest to the observation
		double bestSqDist=DBL_MAX;
		for (unsigned iLandmark=0; iLandmark< landmarks.size(); ++iLandmark){
			// Compute the square of the landmark-observation distance
			auto sqDist= pow((observation.x-landmarks[iLandmark].x_f),2)+pow((observation.y-landmarks[iLandmark].y_f),2);
			// Keep track of the closest match so far
			if (sqDist < bestSqDist) {
				bestSqDist=sqDist;
				observation.id=iLandmark;
			}
		}
		// assert(observation.id!=-1);  // If no measurement of landmark if within sensor range, we are in troubles!
	}
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

	// auto covariance=MatrixXd(2,2);
	// covariance << std_landmark[0], 0, 0, std_landmark[1];
	// MatrixXd invCovariance = covariance.inverse();

	double weightsSum=0;

	for(auto & particle: particles) {
		// Compile list of observations that fall within sensor range. TODO move to main.cpp and do it right!
		/*vector<LandmarkObs> obsInRange;
		for (auto obs: observations) {
			double dist= sqrt(pow(obs.x-particle.x,2)+pow(obs.y-particle.y,2));
			if (dist<sensor_range)
				obsInRange.push_back(obs);
		}*/

		/*vector<Map::single_landmark_s> landmarksInRange;
		for (auto landmark: map_landmarks.landmark_list) {
			double dist= sqrt(pow(landmark.x_f-particle.x,2)+pow(landmark.y_f-particle.y,2));
			if (dist<sensor_range)
				landmarksInRange.push_back(landmark);
		}*/

		// Recompute the observations in the particle reference system, and write them in observationsGlobalRef
		vector<LandmarkObs> observationsGlobalRef=convertLocalToGlobal(particle.x, particle.y, particle.theta, observations);

		matchObservationsWithLandmarks(observationsGlobalRef, map_landmarks.landmark_list);

		// Compute the weight for the given particle
		double weight=1;
		// Empty visualisation and debugging information, to fill it in below
		particle.associations.clear();
		particle.sense_x.clear();
		particle.sense_y.clear();
		for (auto observation: observationsGlobalRef) {
			auto mu_x= map_landmarks.landmark_list[observation.id].x_f;
			auto mu_y= map_landmarks.landmark_list[observation.id].y_f;
			auto sigma_x = std_landmark[0];
			auto sigma_y = std_landmark[1];
			weight*= exp(-(pow(observation.x-mu_x,2)/(2*pow(sigma_x,2))+pow(observation.y-mu_y,2)/(2*pow(sigma_y,2))))/(2*M_PI*sigma_x*sigma_y);
			// Update visualisation and debugging information
			particle.associations.push_back(map_landmarks.landmark_list[observation.id].id_i);
			particle.sense_x.push_back(observation.x);
			particle.sense_y.push_back(observation.y);
		}
		particle.weight = weight;
		weightsSum+=weight;
	}

	// Now all particles have the weights updated, but they still need to be normalised (ensure they add up to 1)
	/*for (auto & particle: particles) {
		particle.weight/=weightsSum;
	}*/
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	vector<Particle> resampled;
	vector<double> weights;
	for (auto particle: particles)
		weights.push_back(particle.weight);
	default_random_engine gen;
	discrete_distribution<> dist(begin(weights), end(weights));
	for (auto count=0; count< num_particles; ++count) {
		auto randomIndex= dist(gen);
		assert(randomIndex>=0);
		assert(static_cast<unsigned>(randomIndex)<particles.size());
		resampled.push_back(particles[randomIndex]);
	}
	particles=resampled;
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
