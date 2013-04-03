class GaussianFilter : public Kernel<uchar4> {
  private:
    Accessor<uchar4> &Input;
    Mask<float> &cMask;

  public:
    GaussianFilter(IterationSpace<uchar4> &IS, Accessor<uchar4> &Input, Mask<float> &cMask) :
      Kernel(IS),
      Input(Input),
      cMask(cMask)
    { addAccessor(&Input); }

    void kernel() {
      output() = convert_uchar4(convolve(cMask, HipaccSUM, [&] () -> float4 {
            return cMask() * convert_float4(Input(cMask));
            }));
    }
};
