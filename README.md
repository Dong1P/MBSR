We plan to upload 'learned network for NTIRE competition (CVPRW 2018)' to other places due to capacity.
# Title : Efficient Module Based SR for multiple problems 
# EMBSR
If you want to run our code, we recommend two ways.
# First
- Dependencies

Python (Tested with 3.6)
PyTorch >= 0.3.1

- Test
If you want to run the code, follow the steps below.

- 0.
track1. bicubic x8.
The order of the test code files is.
a). code_8_4.
b). code 4_2_Final.
c). code 2_1_Final.

track2. mild x4
The order of the test code files is
a). code_4_4_DnCNN_tack2_BN_Resnet.
b). code 4_2_Final_2.
c). code 2_1_Fianl_2.

track3. difficult x4
The order of the test code files is
a). code_4_4_DnCNN_track3_BN_Resnet_2.
b). code 4_2_Final_3.
c). code 2_1_Final_3.

- 1. Please (appath)set the data location in codex_x/data/MyImage.py.
- 2. Modify dir in save_results function in codex_x/utils.py.

# Second
- EMBSR-PyTorch
Experiment can get down here.
# We will update this part soon.


This repository is the code for ‘Efficient Module based on Single Image Super Resolution tackling multiple problems’
- Test Method
Unzip the bicubic, mild, and difficult files and then enter code_2_1 in each folder. Then go into data and modify the location in MyImage.py.

1. Execution
 - Because our method is based on Module, you should train each module for train or test.

1) Testing

First, you should change the path of your data in /data/MyImage. Place your images in test folder. Test results are saved on /experiment/test/results. 

As the method is modeled based approach, you should sequentially operate each module network.
For example, in case of bicubic x8 problem, the sequence is x8 -> x4 ->x2 ->x1(SR).
Firstly, you should make x4(SR) images from x8 images in the code_8_4
Second, you make x2 images from x4(SR) in the code_4_2
Finally, you make x1(SR) images form x2 images(SR) in the code_2_1.

For exception,

cd code_*       # You are now in */EMBSR-PyTorch/code
sh demo.sh


# License
You may freely use and distribute this software as long as you retain the author's name (myself and/or my students) with the software.
It would also be courteous for you to cite the toolbox and any related publications in any papers that present results based on this software. A typical citation is like: Dongwon park, "MBSR," available at [url], downloaded [date].
UM and the authors make all the usual disclaimers about liability etc.
If you make changes to any files, then please change the file name before redistributing to avoid confusion (like the GNU software license). Better yet, email me the changes and I'll consider incorporating them into the toolbox.


We get a lot of help from the site below.

https://github.com/thstkdgus35/EDSR-PyTorch
# Reference
[1] Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee, **"Enhanced Deep Residual Networks for Single Image Super-Resolution,"** <i>2nd NTIRE: New Trends in Image Restoration and Enhancement workshop and challenge on image super-resolution in conjunction with **CVPR 2017**. </i> [[PDF](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Lim_Enhanced_Deep_Residual_CVPR_2017_paper.pdf)] [[arXiv](https://arxiv.org/abs/1707.02921)] [[Slide](https://cv.snu.ac.kr/research/EDSR/Presentation_v3(release).pptx)]
```
@InProceedings{Lim_2017_CVPR_Workshops,
  author = {Lim, Bee and Son, Sanghyun and Kim, Heewon and Nah, Seungjun and Lee, Kyoung Mu},
  title = {Enhanced Deep Residual Networks for Single Image Super-Resolution},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {July},
  year = {2017}
}
