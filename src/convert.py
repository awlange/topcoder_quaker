# Convert neural network JSON to C++ code

import json
import sys


def read_data(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    return data


def main():
    data = read_data(sys.argv[1])

    biases = data.get("biases")
    for i_layer, layer_biases in enumerate(biases):
        s = "      double layer{0}_biases[] = {{".format(i_layer)
        for j_node, node_biases in enumerate(layer_biases):
            for bias in node_biases:
                s += "{0:.6g},".format(bias)
        s = s[:-1] + "};"
        print(s)
    for i_layer, layer_biases in enumerate(biases):
        print("      network.set_layer_biases({0}, layer{0}_biases);".format(i_layer))

    weights = data.get("weights")
    for i_layer, layer_weights in enumerate(weights):
        s = "      double layer{0}_weights[][{1}] = {{\n".format(i_layer, len(layer_weights[0]))
        for j_node, node_weights in enumerate(layer_weights):
            s += "        {"
            for wt in node_weights:
                s += "{0:.6g},".format(wt)
            s = s[:-1] + "},\n"
        s = s[:-2] + "\n      };"
        print(s)

    for i_layer, layer_weights in enumerate(weights):
        print("      for (int j_node = 0; j_node < {0}; j_node++) {{\n".format(len(layer_weights)) +
              "        network.set_layer_node_weights({0}, j_node, {1}, layer{0}_weights[j_node]);\n".format(i_layer, len(layer_weights[0])) +
              "      }"
              )

if __name__ == "__main__":
    main()