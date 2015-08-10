#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>

// Not needed for submission
#include <fstream>
#include <ctime>

using namespace std;

class Network {
  /* --- Members --- */
  private:
    int n_layers;
    vector<int> sizes;
    vector< vector<double> > biases;
    vector< vector< vector<double> > > weights;

  public:

  /* --- Methods --- */

  ~Network() {};

  Network() {
    n_layers = 3;
    int hard_sizes[] = {392, 45, 1393};
    sizes.assign(hard_sizes, hard_sizes+n_layers);

    // Initialize the biases and weights all to 0.0 to start
    int i_layer, j_node, k;
    for (i_layer = 0; i_layer < n_layers-1; i_layer++) {
      vector<double> node_biases;
      vector< vector<double> > node_weights;
      for (j_node = 0; j_node < sizes[i_layer+1]; j_node++) {
        node_biases.push_back(0.0);
        vector<double> k_weights;
        for (k = 0; k < sizes[i_layer]; k++) {
          k_weights.push_back(0.0);
        }
        node_weights.push_back(k_weights);
      }
      biases.push_back(node_biases);
      weights.push_back(node_weights);
    }
  }

  void set_layer_biases(int i_layer, double* b) {
    biases[i_layer].assign(b, b+sizes[i_layer]);
  }

  void set_layer_node_weights(int i_layer, int j_node, int size, double* w) {
    weights[i_layer][j_node].assign(w, w+size);
  }

  /*
   *  Feed forward computation for neural network
   */
  vector<double> feedforward(vector<double> input) {
    vector<double> a = input;
    for (int i_layer = 0; i_layer < sizes.size()-1; i_layer++) {
        vector<double> b;
        vector< vector<double> > layer_weights = weights[i_layer];
        vector<double> layer_biases = biases[i_layer];
        for (int j_node = 0; j_node < sizes[i_layer+1]; j_node++) {
            double node_bias = layer_biases[j_node];
            vector<double> node_weights = layer_weights[j_node];
            double dot = 0.0;
            for (int k = 0; k < node_weights.size(); k++) {
                dot += node_weights[k] * a[k];
            }
            double z = dot + node_bias;
            double sig = 1.0/(1.0 + exp(-z));
            b.push_back(sig);
        }
        a = b;
     }
     return a;
  }


    /*
   *  Feed forward computation for neural network, soft-max output
   */
     /*
  vector<double> feedforward(vector<double> input) {
    vector<double> a = input;
    for (int i_layer = 0; i_layer < sizes.size()-1; i_layer++) {
        vector<double> b;
        vector< vector<double> > layer_weights = weights[i_layer];
        vector<double> layer_biases = biases[i_layer];
        for (int j_node = 0; j_node < sizes[i_layer+1]; j_node++) {
            double node_bias = layer_biases[j_node];
            vector<double> node_weights = layer_weights[j_node];
            double dot = 0.0;
            for (int k = 0; k < node_weights.size(); k++) {
                dot += node_weights[k] * a[k];
            }
            double z = dot + node_bias;
            double sig = 1.0/(1.0 + exp(-z));
            b.push_back(sig);
        }
        a = b;
     }
     // Last layer is soft-max
     for (int i_layer = sizes.size()-1; i_layer < sizes.size()-2; i_layer++) {
        vector<double> b;
        vector< vector<double> > layer_weights = weights[i_layer];
        vector<double> layer_biases = biases[i_layer];
        double soft_denom = 0.0;
        for (int j_node = 0; j_node < sizes[i_layer+1]; j_node++) {
            double node_bias = layer_biases[j_node];
            vector<double> node_weights = layer_weights[j_node];
            double dot = 0.0;
            for (int k = 0; k < node_weights.size(); k++) {
                dot += node_weights[k] * a[k];
            }
            double z = dot + node_bias;
            double soft = exp(z);
            soft_denom += soft;
            b.push_back(soft);
        }
        for (int i = 0; i < b.size(); i++) {
          b[i] /= soft_denom;
        }
        a = b;
     }

     return a;
  }
     */

  /*
   * Static functions to get networks
   */
  static Network getNetwork(int n) {
    switch (n) {
      case 0:
        return getNetwork0();
      default:
        return getNetwork0();
    }
  }

    static Network getNetwork0() {
      Network network;
