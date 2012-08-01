class GaussianFilter : public Kernel<float> {
  private:
    Accessor<float> &Input;
    Mask<float> &cMask;
    const int size_x, size_y;

  public:
    GaussianFilter(IterationSpace<float> &IS, Accessor<float> &Input, Mask<float> &cMask, const int size_x, const int size_y) :
      Kernel(IS),
      Input(Input),
      cMask(cMask),
      size_x(size_x),
      size_y(size_y)
    {
      addAccessor(&Input);
    }

    void kernel() {
      const int ax = size_x >> 1;
      const int ay = size_y >> 1;
      float sum = 0.0f;

      for (int yf = -ay; yf<=ay; yf++) {
        for (int xf = -ax; xf<=ax; xf++) {
          sum += cMask(xf, yf)*Input(xf, yf);
        }
      }
      output() = sum;
    }
};
