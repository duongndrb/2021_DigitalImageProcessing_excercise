clear all; close all; clc


imdata = imread('D:\2021_work\2021_university\2021_03_Image\21_05_folder_image_vision\image100x100.jpeg');

figure(1);imshow(imdata); title('Original Image');

imdata = rgb2gray(imdata);
figure(2); imshow(imdata); title('Gray Image');

%Get Fast Fourier Transform of an image
F = fft2(imdata);
% Fourier Transform of an image
S = abs(F)
figure(3); imshow(S,[]); title('Fourier Transform of an image');

%get the centered spectrum
Fsh = fftshift(F);
figure(4); imshow(abs(Fsh),[]); title('Centered Fourier Transform of Image')

%apply log transform 
S2 = log(1+abs(Fsh));
figure(5); imshow(S2,[]);title('Log transform Image')

%reconstruct the Image
F = ifftshift(Fsh);
f = ifft2(F);
figure(6), imshow(f,[]),title('Reconstructed Image')

% %defing the filter, Low pass filter

[r,c] = size(imdata);
orgr = r/2;
orgc= c/2;
mf = zeros(r,c);
D0= 10;
for i=1:r
    for j=1:c
        if((i-orgr)^2+(j-orgc)^2)^(0.5)<=D0
            mf(j,j) =1
        end
    end
end
figure(7);imshow(uint8(255*mf));title('frequency domain low-past-filter used');

I=Fsh*mf;
figure(8);
I1=log(1+abs(I));
imshow(mat2gray(I1));title('filtered image in frequency domain');
I2=ifft2(ifftshift(I));
figure(9);imshow(uint8(abs(I2)));title('filtered gray scale image');


