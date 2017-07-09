#pragma once

#include "helper_functions.h"
#include <vector>
#include <random>
using std::vector;

struct Particle {

	int id;
	double x;
	double y;
	double theta;
	double weight;
	std::vector<int> associations;
	std::vector<double> sense_x;
	std::vector<double> sense_y;
};

class ParticleFilter {

	// Number of filter particles
	int numParticles;

	// Set to true iff the filter has been initialised, i.e. initialise() has been called
	bool isInitialized;

	// The current filter particles
	std::vector<Particle> particles;

	// Random number generator engine, to be used by this class methods
	std::default_random_engine randomGenerator;

	/**
	 * Convert landmark observations (measurments) from the particle reference system to the global
	 * reference system.
	 * @param x particle x coordinate in the global measurement system [m]
	 * @param y particle y coordinate in the global measurement system [m]
	 * @param theta particle yaw (heading) in the global measurement system [rad]; theta=0 is the orientation of the x axis,
	 * positive angles counter-clockwise
	 * @param observations vector of observations in the given particle reference system; in that system, the x axis is
	 * oriented with the particle yaw (heading)
	 * @return a vector of observations converted to the global reference system
	 */
	vector<LandmarkObs> convertLocalToGlobal(const double x, const double y, const double theta, vector<LandmarkObs> observations);

	/**
	 * Determines for every observation what is the landmark being observed.
	 * For a given observation, the landmark of choice is the one closest to the observed position.
	 * @param observationsGlobalRef a vector of observations expressed in the global reference system (not
	 * the particle reference system); on return the id data member of every observation object
	 * is set to the associated landmark index; the index is the position of the landmark in the given vector of landmarks, numbered
	 * from 0, and it is not the id_i data memebr of the landmark object.
	 * @param landmarks a vector of landmarks whose coordinates are expressed in the global reference system.
	 */
	void matchObservationsWithLandmarks(vector<LandmarkObs> & observationsGlobalRef, const vector<Map::single_landmark_s> landmarks);



public:

	ParticleFilter() :
			numParticles(0), isInitialized(false) {
	}

	~ParticleFilter() {
	}

	/**
	 * Initialises the particle filter; it draws the requested number of particles from the given Gaussian
	 * distribution and sets their weight to 1.
	 * @param x Initial x position [m] (simulated estimate from GPS)
	 * @param y Initial y position [m] (simulated estimate from GPS)
	 * @param theta Initial orientation [rad]
	 * @param std[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 * @param number of particles in the filter
	 */
	void initialise(double x, double y, double theta, double std[], unsigned numParticles);

	/**
	 * Fetches the particle with highest weight from the current filter particles.
	 * @return a copy of the particle with highest weight
	 */
	Particle getBestParticle() const;

	/**
	 * Performs the prediction step of the particle filter, and update the particles
	 * pose, by applying the process model.
	 * @param delta_t Time interval between subsequent measurements [s]
	 * @param std_pos[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 * @param velocity Velocity of car from t to t+delta_t [m/s]
	 * @param yaw_rate Yaw rate of car from t to t+delta_t [rad/s]
	 */
	void predict(double delta_t, double std_pos[], double velocity,
			double yaw_rate);

	/**
	 * Updates the weights for each particle based on the likelihood of the
	 *   observed measurements given that particle. Measurements are bearing
	 *   and distance from landmarks in a map. When updating weights of a
	 *   particle, only landmarks whose distance from the particle doesn't exceed the
	 *   sensor range are considered.
	 * @param sensor_range Range [m] of sensor
	 * @param std_landmark[] Array of dimension 2 [standard deviation of range [m],
	 *   standard deviation of bearing [rad]]
	 * @param observations Vector of landmark observations
	 * @param map Map object containing map landmarks
	 */
	void updateWeights(double sensor_range, double std_landmark[],
			std::vector<LandmarkObs> observations, Map map_landmarks);

	/**
	 * Updates the set of filter particles by sampling it with replacement based on particle weights.
	 * The number of particles in the filter remains unchanged.
	 */
	void resample();

	std::string getAssociations(Particle best);
	std::string getSenseX(Particle best);
	std::string getSenseY(Particle best);

	/**
	 * Returns whether particle filter is initialized yet or not.
	 */
	const bool initialized() const {
		return isInitialized;
	}
};
