class GaussianFilter : public Kernel<float> {
  private:
    Accessor<float> &Input;
    const int size_x, size_y;

  public:
    GaussianFilter(IterationSpace<float> &IS, Accessor<float> &Input, const int size_x, const int size_y) :
      Kernel(IS),
      Input(Input),
      size_x(size_x),
      size_y(size_y)
    { addAccessor(&Input); }

    void kernel() {
      const int ax = size_x >> 1;
      const int ay = size_y >> 1;
      float sum = 0;

      for (int yf = -ay; yf<=ay; yf++) {
        for (int xf = -ax; xf<=ax; xf++) {
          float gauss_constant = expf(-1.0f*((xf*xf)/(2.0f*size_x*size_x) + (yf*yf)/(2.0f*size_y*size_y)));
          sum += gauss_constant*Input(xf, yf);
        }
      }
      output() = sum;
    }
};
