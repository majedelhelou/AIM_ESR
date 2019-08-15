hrdir = '../dataset/trainHR/';
outputlrfile = '../dataset/lr.h5'
outputhrfile = '../dataset/hr.h5'
psize = 32;
phrsize = 32*16;

s = 32;
shr = 32*16;

fnum = 1;
files = dir([hrdir '*.png']);
h5create(outputlrfile, '/data', [psize psize 3 Inf], 'Datatype', 'single', 'ChunkSize', [psize psize 3 1]);
h5create(outputhrfile, '/data', [phrsize phrsize 3 Inf], 'Datatype', 'single', 'ChunkSize', [phrsize phrsize 3 1]);
for k = 1:length(files)
    file = [hrdir files(k).name];
    hrI = imread(file);
    lrI = imresize(hrI, 1.0/16.0);
    [lrw, lrh, lrc] = size(lrI);
    [hrw, hrh, hrc] = size(hrI);
    hi = 1;
    
    for lri = 1:s:lrw
        hj = 1;
        lrj = 1;
        if ((lri + psize > lrw) | (hi + phrsize > hrw))
               break;
        end
        
        for lrj = 1:s:lrh
            if ((lrj + psize > lrh) | (hj + phrsize> hrh))
                continue;
            end
            hrpatch = hrI(hi:hi+phrsize-1, hj:hj+phrsize-1, :);
            lrpatch = lrI(lri:lri+psize-1, lrj:lrj+psize-1, :);
            
            h5write(outputlrfile, '/data', single(lrpatch),  [1 1,1,fnum], [psize, psize, 3 1]);
            h5write(outputhrfile, '/data', single(hrpatch),  [1,1,1,fnum], [phrsize, phrsize, 3 1]);
            fnum = fnum + 1;
            if (mod(fnum,100) == 0)
                sprintf('already generated %d patches\n', fnum)
            end
            lrj = lrj + s;
            hj = hj + shr;
        end
        hi = hi + shr;
        lri = lri + s;
    end
end
sprintf('generated %d patches\n', fnum)
