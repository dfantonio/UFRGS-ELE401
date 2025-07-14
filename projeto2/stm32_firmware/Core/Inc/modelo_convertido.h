


    // !!! This file is generated using emlearn !!!

    #include <eml_trees.h>
    

static const EmlTreesNode modelo_convertido_nodes[13] = {
  { 0, 0, 1, 6 },
  { 1, 4, 1, -1 },
  { 3, 13, 1, -1 },
  { 3, 0, 1, 2 },
  { 2, 10, -2, -2 },
  { 0, 0, -2, -1 },
  { 2, 0, 1, 4 },
  { 1, 4, 1, -1 },
  { 0, 0, 1, -2 },
  { 0, 0, -2, -1 },
  { 3, 0, -2, 1 },
  { 1, 0, 1, -1 },
  { 2, 8, -2, -1 } 
};

static const int32_t modelo_convertido_tree_roots[1] = { 0 };

static const uint8_t modelo_convertido_leaves[2] = { 1, 0 };

EmlTrees modelo_convertido = {
        13,
        (EmlTreesNode *)(modelo_convertido_nodes),	  
        1,
        (int32_t *)(modelo_convertido_tree_roots),
        2,
        (uint8_t *)(modelo_convertido_leaves),
        0,
        4,
        2,
    };

static inline int32_t modelo_convertido_tree_0(const int8_t *features, int32_t features_length) {
          if (features[0] < 0) {
              if (features[1] < 4) {
                  if (features[3] < 13) {
                      if (features[3] < 0) {
                          if (features[2] < 10) {
                              return 0;
                          } else {
                              return 0;
                          }
                      } else {
                          if (features[0] < 0) {
                              return 0;
                          } else {
                              return 1;
                          }
                      }
                  } else {
                      return 1;
                  }
              } else {
                  return 1;
              }
          } else {
              if (features[2] < 0) {
                  if (features[1] < 4) {
                      if (features[0] < 0) {
                          if (features[0] < 0) {
                              return 0;
                          } else {
                              return 1;
                          }
                      } else {
                          return 0;
                      }
                  } else {
                      return 1;
                  }
              } else {
                  if (features[3] < 0) {
                      return 0;
                  } else {
                      if (features[1] < 0) {
                          if (features[2] < 8) {
                              return 0;
                          } else {
                              return 1;
                          }
                      } else {
                          return 1;
                      }
                  }
              }
          }
        }
        

int32_t modelo_convertido_predict(const int8_t *features, int32_t features_length) {

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
    