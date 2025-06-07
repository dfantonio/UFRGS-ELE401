


    // !!! This file is generated using emlearn !!!

    #include <eml_trees.h>
    

static const EmlTreesNode modelo_convertido_nodes[14] = {
  { 8, -0.089096f, 1, 9 },
  { 21, 0.307685f, 1, -1 },
  { 19, -1.432577f, -1, 1 },
  { 14, 0.026356f, 1, 5 },
  { 20, -0.938216f, 1, 2 },
  { 15, -0.961865f, -1, -2 },
  { 22, 1.236437f, -2, 1 },
  { 0, -0.235981f, -1, -2 },
  { 30, -0.698562f, -2, -1 },
  { 22, -0.448182f, 1, 3 },
  { 24, 0.130027f, 1, -1 },
  { 0, -0.236257f, -1, -2 },
  { 14, -0.591012f, -2, 1 },
  { 23, -0.515629f, -2, -1 } 
};

static const int32_t modelo_convertido_tree_roots[1] = { 0 };

static const uint8_t modelo_convertido_leaves[2] = { 1, 0 };

EmlTrees modelo_convertido = {
        14,
        (EmlTreesNode *)(modelo_convertido_nodes),	  
        1,
        (int32_t *)(modelo_convertido_tree_roots),
        2,
        (uint8_t *)(modelo_convertido_leaves),
        0,
        31,
        2,
    };

static inline int32_t modelo_convertido_tree_0(const float *features, int32_t features_length) {
          if (features[8] < -0.089096f) {
              if (features[21] < 0.307685f) {
                  if (features[19] < -1.432577f) {
                      return 1;
                  } else {
                      if (features[14] < 0.026356f) {
                          if (features[20] < -0.938216f) {
                              if (features[15] < -0.961865f) {
                                  return 1;
                              } else {
                                  return 0;
                              }
                          } else {
                              if (features[22] < 1.236437f) {
                                  return 0;
                              } else {
                                  if (features[0] < -0.235981f) {
                                      return 1;
                                  } else {
                                      return 0;
                                  }
                              }
                          }
                      } else {
                          if (features[30] < -0.698562f) {
                              return 0;
                          } else {
                              return 1;
                          }
                      }
                  }
              } else {
                  return 1;
              }
          } else {
              if (features[22] < -0.448182f) {
                  if (features[24] < 0.130027f) {
                      if (features[0] < -0.236257f) {
                          return 1;
                      } else {
                          return 0;
                      }
                  } else {
                      return 1;
                  }
              } else {
                  if (features[14] < -0.591012f) {
                      return 0;
                  } else {
                      if (features[23] < -0.515629f) {
                          return 0;
                      } else {
                          return 1;
                      }
                  }
              }
          }
        }
        

int32_t modelo_convertido_predict(const float *features, int32_t features_length) {

        int32_t votes[2] = {0,};
        int32_t _class = -1;

        _class = modelo_convertido_tree_0(features, features_length); votes[_class] += 1;
    
        int32_t most_voted_class = -1;
        int32_t most_voted_votes = 0;
        for (int32_t i=0; i<2; i++) {

            if (votes[i] > most_voted_votes) {
                most_voted_class = i;
                most_voted_votes = votes[i];
            }
        }
        return most_voted_class;
    }
    