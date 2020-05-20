function r_map=average_energy_1(band_filter,J,row,col)
%E=0;
for i=3:row+2
    for j=3:col+2
        pixel=0;
        for k = -2:2
            for l = -2:2
                pixel = pixel + (J(i+k,j+l)*band_filter(k+3,l+3));
            end
        end
        r_map(i-2,j-2)=pixel;
        %E = E + ((1/(row*col))*(pixel));
    end
end
end