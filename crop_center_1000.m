function Ic=crop_center_1000(I)
    sz = size(I);
    Ic = imcrop(I, [sz(2)/2-499 sz(1)/2-499 999 999]);    
end