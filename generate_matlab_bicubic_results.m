input_datapath = 'dataset/validationLR/';
output_datapath = 'results/matlab_bicubic/';

imgs = dir(strcat(input_datapath, '*.png'));
for i = 1:numel(imgs)
    img_file = strcat(input_datapath , imgs(i).name);
    I = imread(img_file);
    I = imresize(I, 16);
    Ic = crop_center_1000(I);
    imwrite(Ic, strcat(output_datapath, imgs(i).name))
end