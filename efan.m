function out = efan(img)

    s = size(img);
    dims = size(s);
    img = double(img);
    %if size(img,3)==3
    if dims(2) == 3
        [ht wd dt] = size(img);
        M = zeros(ht,wd);
        M(1:16:end,1:16:end) = 1;
        sz= wd*ht;
        
        r = (img(:,:,1)).*M;
        g = (img(:,:,2)).*M;
        b = (img(:,:,3)).*M;

        numrandom = sum(sum(M));
        percentage = double(numrandom)/double(sz);
        %sigma = sqrt(1.0/percentage);
        sigma = sqrt(double(sz)/(pi*double(numrandom)));
        
        sigma = sigma*2;

        r = fftfilt(r,sigma);
        g = fftfilt(g,sigma);
        b = fftfilt(b,sigma);
        i = fftfilt(M,sigma);

        r = r./i;
        g = g./i;
        b = b./i;

        out = img;
        out(:,:,1) = uint8(r);
        out(:,:,2) = uint8(g);
        out(:,:,3) = uint8(b);

    else
        [ht wd] = size(img);
        sz= wd*ht;
        g = (img).*M;
        
        numrandom = sum(sum(M));
        percentage = double(numrandom)/double(sz);
        %sigma = sqrt(1.0/percentage);
        sigma = sqrt(double(sz)/(pi*double(numrandom)));        
        
        g = fftfilt(g,sigma);
        i = fftfilt(M,sigma);
        g = g./i;

        out = uint8(g);

    end
end

function out = fftfilt(in, sigma)
    [ht,wd] = size(in);
    [X,Y] = meshgrid(1:wd,1:ht);
    X = X-wd/2;
    Y = Y-ht/2;
    Z = exp(-0.5*(X.^2 + Y.^2)/sigma^2);
    Z = Z/sum(sum(Z));
    out = ifftshift(ifft2( fft2(in).*fft2(Z) ));
    %figure, imshow(uint8(out));
end