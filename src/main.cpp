#include <uWS/uWS.h>
#include <iostream>
#include "json.hpp"
#include <math.h>
#include "particle_filter.h"

using namespace std;

// for convenience
using json = nlohmann::json;

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
std::string hasData(std::string s) {
	auto found_null = s.find("null");
	auto b1 = s.find_first_of("[");
	auto b2 = s.find_first_of("]");
	if (found_null != std::string::npos) {
		return "";
	} else if (b1 != std::string::npos && b2 != std::string::npos) {
		return s.substr(b1, b2 - b1 + 1);
	}
	return "";
}

bool near(double a,double b) {
	double epsilon = 0.000001;
	return (abs(a-b)< epsilon)? true: false;
}
/*
void test() {
	ParticleFilter pf;
	pf.testInit();
	double sigma_pos[3] = { 0, 0, 0};
	pf.predict(0.1, sigma_pos, 10, 0);
	assert(pf.particles.size()==2);
	assert(near(pf.particles[0].theta,0));
	assert(near(pf.particles[0].x,11));
	assert(near(pf.particles[0].y,20));
	assert(near(pf.particles[1].theta,M_PI/2));
	assert(near(pf.particles[1].x,20));
	assert(near(pf.particles[1].y,11));

	double sensor_range=50;
	double sigma_landmark[2] = { 0.3, 0.3 };
	Map map;
	map.landmark_list = vector<Map::single_landmark_s>(3);
	map.landmark_list[0].id_i=1;
	map.landmark_list[0].x_f=0;
	map.landmark_list[0].y_f=0;
	map.landmark_list[1].id_i=2;
	map.landmark_list[1].x_f=30;
	map.landmark_list[1].y_f=30;
	map.landmark_list[2].id_i=3;
	map.landmark_list[2].x_f=100;
	map.landmark_list[2].y_f=100;

	std::vector<LandmarkObs> observations(3);
	observations[0].id=-1;
	observations[0].x=-10;
	observations[0].y=-20;
	observations[1].id=-1;
	observations[1].x=20;
	observations[1].y=10;
	observations[2].id=-1;
	observations[2].x=90;
	observations[2].y=80;

	pf.updateWeights(sensor_range, sigma_landmark, observations, map);
	cout << "Unit test done" << endl;
	exit(0);

}*/
int main() {
	//test();
	uWS::Hub h;

	//Set up parameters here
	double deltaT = 0.1; // Time elapsed between measurements [sec]
	double sensorRange = 50; // Sensor range [m]

	double sigmaPos[3] = { 0.3, 0.3, 0.01 }; // GPS measurement uncertainty [x [m], y [m], theta [rad]]
	double sigmaLandmarks[2] = { 0.3, 0.3 }; // Landmark measurement uncertainty [x [m], y [m]]

	// Read map data
	Map map;
	if (!readMapData("../data/map_data.txt", map)) { // TODO change this to "../data/map_data.txt" before handing it in!
		cout << "Error: Could not open map file" << endl;
		return -1;
	}

	// Create particle filter
	ParticleFilter pf;

	h.onMessage(
			[&pf,&map,&deltaT,&sensorRange,&sigmaPos,&sigmaLandmarks](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode opCode) {
				// "42" at the start of the message means there's a websocket message event.
				// The 4 signifies a websocket message
				// The 2 signifies a websocket event

				if (length && length > 2 && data[0] == '4' && data[1] == '2')
				{

					auto s = hasData(std::string(data));
					if (s != "") {

						auto j = json::parse(s);
						std::string event = j[0].get<std::string>();

						if (event == "telemetry") {
							// j[1] is the data JSON object

							if (!pf.initialized()) {

								// Sense noisy position data from the simulator
								double sense_x = std::stod(j[1]["sense_x"].get<std::string>());
								double sense_y = std::stod(j[1]["sense_y"].get<std::string>());
								double sense_theta = std::stod(j[1]["sense_theta"].get<std::string>());

								pf.initialise(sense_x, sense_y, sense_theta, sigmaPos, 100);
							}
							else {
								// Predict the vehicle's next state from previous (noiseless control) data.
								double previous_velocity = std::stod(j[1]["previous_velocity"].get<std::string>());
								double previous_yawrate = std::stod(j[1]["previous_yawrate"].get<std::string>());

								pf.predict(deltaT, sigmaPos, previous_velocity, previous_yawrate);
							}

							// receive noisy observation data from the simulator
							// sense_observations in JSON format [{obs_x,obs_y},{obs_x,obs_y},...{obs_x,obs_y}]
							vector<LandmarkObs> noisy_observations;
							string sense_observations_x = j[1]["sense_observations_x"];
							string sense_observations_y = j[1]["sense_observations_y"];

							std::vector<float> x_sense;
							std::istringstream iss_x(sense_observations_x);

							std::copy(std::istream_iterator<float>(iss_x),
									std::istream_iterator<float>(),
									std::back_inserter(x_sense));

							std::vector<float> y_sense;
							std::istringstream iss_y(sense_observations_y);

							std::copy(std::istream_iterator<float>(iss_y),
									std::istream_iterator<float>(),
									std::back_inserter(y_sense));

							for(unsigned i = 0; i < x_sense.size(); i++)
							{
								LandmarkObs obs;
								obs.x = x_sense[i];
								obs.y = y_sense[i];
								noisy_observations.push_back(obs);
							}

							// Update the weights and resample
							pf.updateWeights(sensorRange, sigmaLandmarks, noisy_observations, map);
							pf.resample();

							auto bestParticle = pf.getBestParticle();
							json msgJson;
							msgJson["best_particle_x"] = bestParticle.x;
							msgJson["best_particle_y"] = bestParticle.y;
							auto theta = bestParticle.theta;
							msgJson["best_particle_theta"] = theta;

							//Optional message data used for debugging particle's sensing and associations
							msgJson["best_particle_associations"] = pf.getAssociations(bestParticle);
							msgJson["best_particle_sense_x"] = pf.getSenseX(bestParticle);
							msgJson["best_particle_sense_y"] = pf.getSenseY(bestParticle);

							auto msg = "42[\"best_particle\"," + msgJson.dump() + "]";
							// std::cout << msg << std::endl;
							ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);

						}
					} else {
						std::string msg = "42[\"manual\",{}]";
						ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
					}
				}

			});

	// We don't need this since we're not using HTTP but if it's removed the program
	// doesn't compile :-(
	h.onHttpRequest(
			[](uWS::HttpResponse *res, uWS::HttpRequest req, char *data, size_t, size_t) {
				const std::string s = "<h1>Hello world!</h1>";
				if (req.getUrl().valueLength == 1)
				{
					res->end(s.data(), s.length());
				}
				else
				{
					// i guess this should be done more gracefully?
					res->end(nullptr, 0);
				}
			});

	h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
		std::cout << "Connected!!!" << std::endl;
	});

	h.onDisconnection(
			[&h](uWS::WebSocket<uWS::SERVER> ws, int code, char *message, size_t length) {
				ws.close();
				std::cout << "Disconnected" << std::endl;
			});

	int port = 4567;
	if (h.listen(port)) {
		std::cout << "Listening to port " << port << std::endl;
	} else {
		std::cerr << "Failed to listen to port" << std::endl;
		return -1;
	}
	h.run();
}

