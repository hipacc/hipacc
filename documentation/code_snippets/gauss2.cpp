class GaussianFilter : public Kernel<float> {
  private:
    Accessor<float> &Input;
    Mask<float> &cMask;

  public:
    GaussianFilter(IterationSpace<float> &IS, Accessor<float> &Input, Mask<float> &cMask) :
      Kernel(IS),
      Input(Input),
      cMask(cMask)
    {
      addAccessor(&Input);
    }

    void kernel() {
      output() = convolve(cMask, SUM, [&] () {
        return cMask()*Input(cMask);
        });
    }
};
