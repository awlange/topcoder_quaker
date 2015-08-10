      return network;
    }
    // banana

};

class QuakePredictor {

  /* --- Members --- */
  private:
    int qp_sampleRate;
    int qp_S;
    vector<double> qp_sitesData;
    vector<double> qp_retM;
    vector<Network> bag_of_networks;

  public:

  /* --- Methods --- */

  QuakePredictor() {};
  ~QuakePredictor() {};

  private:
    /*
     * Discrete Fourier Transform, convert to power spectrum, sum every 18
     */
    vector<double> dft_power(vector<double> data) {
      int n_data = data.size();
      int n_j = n_data / 2;
      vector<double> ft;
      double two_pi_over_n = 2.0 * M_PI / (double)n_data;
      double power_sum = 0.0;
      for (int j = 0; j < n_j; ++j) {
        double j_two_pi_over_n = (double)j * two_pi_over_n;
        double yj_re = 0.0;
        double yj_im = 0.0;
        for (int k = 0; k < n_data; ++k) {
          double z = j_two_pi_over_n * (double)k;
          yj_re += data[k] * cos(z);
          yj_im += data[k] * sin(-z);
        }
        power_sum += sqrt(yj_re*yj_re + yj_im*yj_im);
        if ((j+1) % 18 == 0) {
          ft.push_back(power_sum);
          power_sum = 0.0;
        }
      }
      return ft;
    }

    /*
     This computes an in-place complex-to-complex FFT
     x and y are the real and imaginary arrays of 2^m points.
    */
    short FFT(long m, vector<double> x, vector<double> y) {
      long n,i,i1,j,k,i2,l,l1,l2;
      double c1,c2,tx,ty,t1,t2,u1,u2,z;

      n = 4096;

      /* Do the bit reversal */
      i2 = n >> 1;
      j = 0;
      for (i = 0; i < n-1; i++) {
         if (i < j) {
            tx = x[i];
            x[i] = x[j];
            x[j] = tx;
         }
         k = i2;
         while (k <= j) {
            j -= k;
            k >>= 1;
         }
         j += k;
      }

      /* Compute the FFT */
      c1 = -1.0;
      c2 = 0.0;
      l2 = 1;
      for (l = 0; l < m; l++) {
         l1 = l2;
         l2 <<= 1;
         u1 = 1.0;
         u2 = 0.0;
         for (j = 0; j < l1; j++) {
            for (i = j; i < n; i+= l2) {
               i1 = i + l1;
               t1 = u1 * x[i1] - u2 * y[i1];
               t2 = u1 * y[i1] + u2 * x[i1];
               x[i1] = x[i] - t1;
               y[i1] = y[i] - t2;
               x[i] += t1;
               y[i] += t2;
            }
            z =  u1 * c1 - u2 * c2;
            u2 = u1 * c2 + u2 * c1;
            u1 = z;
         }
         c2 = sqrt((1.0 - c1) / 2.0);
         c2 = -c2;
         c1 = sqrt((1.0 + c1) / 2.0);
      }

      return 1;
    }

    /*
     *  FFT power spectrum. Input is destroyed.
     */
    vector<double> fft_power(vector<double> data) {
      // Zero pad data out to 4096
      int i, j;
      for (i = 0; i < 496; ++i) {
        data.push_back(0.0);
      }

      // Imaginary values
      vector<double> y;
      for (i = 0; i < 4096; ++i) {
        y.push_back(0.0);
      }

      // Run the FFT
      FFT(12, data, y);

      // Integrate every 32 values. But we only need to do half
      vector<double> power;
      j = 0;
      double power_sum = 0.0;
      for (i = 0; i < 2048; ++i) {
        j++;
        power_sum += sqrt(data[i]*data[i] + y[i]*y[i]);
        if (j % 32 == 0) {
          power.push_back(power_sum);
          j = 0;
          power_sum = 0.0;
        }
      }
      return power;
    }

    /*
     * Distance between two sites
     */
    double distance(double lat1, double lon1, double lat2, double lon2) {
      double earthRadius = 6371.01;

      double deltaLon = fabs(lon1 - lon2);
      if (deltaLon > 180.0) {
        deltaLon = 360.0 - deltaLon;
      }

      // Convert to radians
      double deg2rad = M_PI / 180.0;
      deltaLon *= deg2rad;
      lat1 *= deg2rad;
      lat2 *= deg2rad;

      double cl11 = cos(lat1);
      double sl11 = sin(lat1);
      double cl21 = cos(lat2);
      double sl21 = sin(lat2);
      double cdeltaLon = cos(deltaLon);
      double sdeltaLon = sin(deltaLon);

      double tmp1 = cl11 * sdeltaLon;
      double tmp2 = cl21 * sl11 - sl21 * cl11 * cdeltaLon;

      double x = sqrt(tmp1*tmp1 + tmp2*tmp2);
      double y = sl21 * sl11 + cl21 * cl11 * cdeltaLon;
      double dist = earthRadius * atan2(x, y);

      return dist;
    }

  public:
    /*
     * Initialization function
     */
    int init(int sampleRate, int S, vector<double> sitesData) {
      int i;

      qp_sampleRate = sampleRate;
      qp_S = S;
      qp_sitesData = sitesData;

      for (i = 0; i < qp_S * 90 * 24; ++i) {
        qp_retM.push_back(0.0);
      }

      // Initialize the neural network(s)
      for (i = 0; i < 1; ++i) {
        bag_of_networks.push_back(Network::getNetwork(i));
      }

      return 0;
    }

    /*
     * Forecast function
     */
    vector<double> forecast(int hour, vector<int> data, double K, vector<double> globalQuakes) {

      int i, j, k;
      int H = qp_sampleRate;
      int S = qp_S;
      int H3600 = H * 3600;

      // Scale to be in range 0-1
      double max_val = pow(2.0, 24.0);
      vector<double> scaled_data;
      for (i = 0; i < data.size(); ++i) {
        scaled_data.push_back(((double)data[i]) / max_val);
      }
      double scaled_K = (double)K / 10.0;

      // Set all to zero.
      for (i = 0; i < qp_S * 90 * 24; ++i) {
        qp_retM[i] = 0.0;
      }

      // For each site...
      for (j = 0; j < S; ++j) {

        // Data transforms
        vector<double> channel_stats;
        vector<double> second_vals;
        for (int c = 0; c < 3; ++c) {

          // Compute average
          int n = 0;
          double avg = 0.0;
          int offset = (j * 3 * H3600) + (c * H3600);
          for (i = 0; i < H3600; ++i) {
            double val = scaled_data[offset + i];
            if (val >= 0.0) {
              n += 1;
              avg += val;
            }
          }
          if (n > 0) {
            avg /= (double)n;
          }

          // Mean fill missing values
          vector<double> mean_filled_values;
          double mean = 0.0;
          for (i = 0; i < H3600; ++i) {
            double tmp = (scaled_data[offset + i] > 0.0) ? scaled_data[offset + i] : avg;
            mean += tmp;
            mean_filled_values.push_back(tmp);
          }
          mean /= (double)H3600;

          // Compute seconds means, mean center
          i = 0;
          double second_mean = 0.0;
          vector<double> second_means;
          for (int v = 0; v < mean_filled_values.size(); ++v) {
            double val = mean_filled_values[v];
            i += 1;
            second_mean += val - mean;
            if (i == H) {
              second_means.push_back(second_mean / (double)H);
              second_mean = 0.0;
              i = 0;
            }
          }

          // Simple smoothing
          double smooth_stdev = 0.0;
          vector<double> smoothed_vals;
          double vall;
          vall = second_means[3600-2] * 0.05
               + second_means[3600-1] * 0.20
               + second_means[0] * 0.50
               + second_means[1] * 0.20
               + second_means[2] * 0.05;
          smooth_stdev += vall*vall;
          smoothed_vals.push_back(vall);
          vall = second_means[3600-1] * 0.05
               + second_means[0] * 0.20
               + second_means[1] * 0.50
               + second_means[2] * 0.20
               + second_means[3] * 0.05;
          smooth_stdev += vall*vall;
          smoothed_vals.push_back(vall);
          for (i = 2; i < 3600; ++i) {
            vall = second_means[i-2] * 0.05
                 + second_means[i-1] * 0.20
                 + second_means[i] * 0.50
                 + second_means[(i+1) % 3600] * 0.20
                 + second_means[(i+2) % 3600] * 0.05;
            smooth_stdev += vall*vall;
            smoothed_vals.push_back(vall);
          }
          smooth_stdev = sqrt(smooth_stdev / 3600.0);

          // First derivative of smoothed vals
          vector<double> second_1_derivs;
          double fp1 = smoothed_vals[1];
          double fm1 = smoothed_vals[3600-1];
          second_1_derivs.push_back(0.5 * (fp1 - fm1));
          for (i = 1; i < 3600-1; ++i) {
            fp1 = smoothed_vals[i+1];
            fm1 = smoothed_vals[i-1];
            second_1_derivs.push_back(0.5 * (fp1 - fm1));
          }
          fp1 = smoothed_vals[0];
          fm1 = smoothed_vals[3600-2];
          second_1_derivs.push_back(0.5 * (fp1 - fm1));

          // Fourier transforms, power spectra
          vector<double> ft_second_vals = fft_power(smoothed_vals);
          vector<double> ft_second_1_derivs = fft_power(second_1_derivs);

          // Collect up the data
          channel_stats.push_back(mean);
          channel_stats.push_back(smooth_stdev);
          for (i = 0; i < ft_second_vals.size(); ++i) {
            second_vals.push_back(ft_second_vals[i]);
          }
          for (i = 0; i < ft_second_1_derivs.size(); ++i) {
            second_vals.push_back(ft_second_1_derivs[i]);
          }

        }

        // Compute global quake score per site
        double global_quakes_score = 0.0;
        if (globalQuakes.size() > 0) {
          double site_location_lat = qp_sitesData[j*2];
          double site_location_lon = qp_sitesData[j*2+1];
          for (i = 0; i < globalQuakes.size() / 5; ++i) {
            double dist = distance(site_location_lat, site_location_lon,
                                   globalQuakes[i*5], globalQuakes[i*5+1]);
            global_quakes_score += globalQuakes[i*5+3] / dist;
          }
        }

        // Create input vector for networks
        vector<double> input_vector;
        input_vector.push_back(scaled_K);
        input_vector.push_back(global_quakes_score);
        for (i = 0; i < channel_stats.size(); ++i) {
          input_vector.push_back(channel_stats[i]);
        }
        for (i = 0; i < second_vals.size(); ++i) {
          input_vector.push_back(second_vals[i]);
        }

        // Run neural network for prediction. Mean aggregate.
        //double prediction = 0.0;
        //for (i = 0; i < bag_of_networks.size(); ++i) {
        //  prediction += bag_of_networks[i].feedforward(input_vector);
        //}
        //prediction /= (double)bag_of_networks.size();

        // No bag, just one now.
        vector<double> prediction; 
        prediction = bag_of_networks[0].feedforward(input_vector);

        // Store prediction in the return matrix
        //qp_retM[qp_S*hour + j] = prediction;

        // Set values, zero element gets ignored.
        for (i = 1; i < prediction.size(); i++) { 
          int adjusted_hour = hour + i;
          if (adjusted_hour < 2160) {
            qp_retM[qp_S*adjusted_hour + j] = prediction[i]; 
          }
        }

      }

      return qp_retM;
    }

};


/*
 *  The main function
 */
int main(int argc, char** argv) {

  QuakePredictor QP;

  int sampleRate, S, SLEN;
  cin >> sampleRate;
  cin >> S;
  cin >> SLEN;

  vector<double> sitesData;
  for (int s = 0; s < SLEN; s++) {
    double tmp;
    cin >> tmp;
    sitesData.push_back(tmp);
  }

  int ret = QP.init(sampleRate, S, sitesData);
  cout << ret << endl;

  int doTraining = 0;
  cin >> doTraining;
  if (doTraining) {
    // pass
  }

  // Read data for each hour
  while (true) {
    int hour;
    cin >> hour;
    if (hour < 0) {
      break;
    }

    int DLEN;
    cin >> DLEN;
    vector<int> data;
    for (int d = 0; d < DLEN; d++) {
      double tmp;
      cin >> tmp;
      data.push_back(tmp);
    }

    double K;
    cin >> K;

    int QLEN;
    cin >> QLEN;
    vector<double> globalQuakes;
    for (int q = 0; q < QLEN; q++) {
      double tmp;
      cin >> tmp;
      globalQuakes.push_back(tmp);
    }

    // The big forecast call
    vector<double> retM;
    retM = QP.forecast(hour, data, K, globalQuakes);
    cout << retM.size() << endl;
    for (int r = 0; r < retM.size(); r++) {
      cout << retM[r] << endl;
    }
  }

  return 0;
}
