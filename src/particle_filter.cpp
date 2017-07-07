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
	numParticles = 100;
	auto sigma_x = std[0];
	auto sigma_y = std[1];
	auto sigma_theta = std[2];
	default_random_engine gen;
	normal_distribution<double> dist_x(x, sigma_x);
	normal_distribution<double> dist_y(y, sigma_y);
	normal_distribution<double> dist_theta(theta, sigma_theta);
	for (auto count = 0; count < numParticles; ++count) {
		Particle newParticle;
		newParticle.id = count;
		newParticle.x = dist_x(gen);
		newParticle.y = dist_y(gen);
		newParticle.theta = dist_theta(gen);
		newParticle.theta = normaliseAngle(newParticle.theta);
		newParticle.weight = 1./numParticles;
		particles.push_back(newParticle);
	}
	isInitialized = true;
}

void ParticleFilter::testInit() {
	numParticles = 2;
	Particle newParticle1;
	newParticle1.id=1;
	newParticle1.x=10;
	newParticle1.y=20;
	newParticle1.theta=0;
	newParticle1.weight=1./numParticles;
	Particle newParticle2;
	newParticle2.id=2;
	newParticle2.x=20;
	newParticle2.y=10;
	newParticle2.weight=1./numParticles;
	newParticle2.theta=3.1415926535/2;
	particles.push_back(newParticle1);
	particles.push_back(newParticle2);
	isInitialized = true;
}

void ParticleFilter::predict(double delta_t, double std_pos[],
		double velocity, double yaw_rate) {
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
	for(auto & particle: particles) {
		// Recompute the observations in the particle reference system, and write them in observationsGlobalRef
		vector<LandmarkObs> observationsGlobalRef=convertLocalToGlobal(particle.x, particle.y, particle.theta, observations);


		vector<Map::single_landmark_s> landmarksInRange;
		for (auto landmark : map_landmarks.landmark_list)
			if (sqrt(pow(landmark.x_f-particle.x,2)+pow(landmark.y_f-particle.y, 2))<= sensor_range)
				landmarksInRange.push_back(landmark);

		matchObservationsWithLandmarks(observationsGlobalRef, landmarksInRange);

		// Compute the weight for the given particle
		double weight=1;
		// Empty visualisation and debugging information, to fill it in below
		particle.associations.clear();
		particle.sense_x.clear();
		particle.sense_y.clear();
		for (auto observation: observationsGlobalRef) {
			auto mu_x= landmarksInRange[observation.id].x_f;
			auto mu_y= landmarksInRange[observation.id].y_f;
			auto sigma_x = std_landmark[0];
			auto sigma_y = std_landmark[1];
			weight*= exp(-(pow(observation.x-mu_x,2)/(2*pow(sigma_x,2))+pow(observation.y-mu_y,2)/(2*pow(sigma_y,2))))/(2*M_PI*sigma_x*sigma_y);
			// Update visualisation and debugging information
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
	default_random_engine gen;
	discrete_distribution<> dist(begin(weights), end(weights));
	for (auto count=0; count< numParticles; ++count) {
		auto randomIndex= dist(gen);
		assert(randomIndex>=0);
		assert(static_cast<unsigned>(randomIndex)<particles.size());
		resampled.push_back(particles[randomIndex]);
	}
	particles=resampled;
}

Particle ParticleFilter::setAssociations(Particle particle,
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
