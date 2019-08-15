input_datapath = 'dataset/validationLR/';
output_datapath = 'results/efan/';

imgs = dir(strcat(input_datapath, '*.png'));
for i = 1:numel(imgs)
    img_file = strcat(input_datapath , imgs(i).name);
    I = imread(img_file);
    I = imresize(I, 16);
    Iout = uint8(efan(I));
    Ic = crop_center_1000(Iout);
    imwrite(Ic, strcat(output_datapath, imgs(i).name))
end