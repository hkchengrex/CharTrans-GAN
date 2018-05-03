# CharTrans-GAN
Use GAN to perform style transfer of Chinese characters.

*BAD CODE WARNING: * Rushed this project in a short time. Quite naïve and simple. My GPU has been working 24-7 but still I didn't get to try much, so don't actually trust the things I said.

[Report](/report/final/egpaper_final.pdf)

[Presentation](https://docs.google.com/presentation/d/e/2PACX-1vTrG_QY-UH8aeHO-pQqtJnMGw59j05pvyLZ7AkOO_g2-v3smdjlnjk0pJNza_FUY7vn5m1UuKLhk9xl/pub#slide=id.g393abc15cf_0_0)

## Network structure

### Generator
![Generator](/report/final/gen.png)

### Discriminator
![Generator](/report/final/dis.png)

## Some results
| Content Reference | Generated | Ground truth |
| -------- | --------- | --------- |
|![Content](/report/final/380r1.png)|![Gen](/report/final/380gen.png)|![GT](/report/final/380gt.png)|
|![Content](/report/final/385r1.png)|![Gen](/report/final/385gen.png)|![GT](/report/final/385gt.png)|

## Reference
Learnt the implementation of GAN from [pytorch-generative-model-collections](https://github.com/znxlwm/pytorch-generative-model-collections).
