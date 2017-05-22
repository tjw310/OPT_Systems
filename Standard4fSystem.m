classdef Standard4fSystem < OPTSystem
    %Simulation of point objects in cone-beam OPT system
    
    properties (Access = private)
    end
    
    methods %constructor
        function obj = Standard4fSystem()
            obj = obj@OPTSystem();
        end
    end
   
    methods    
        % @param double x,y,z , location of point object
        % @param Objective objective, objective class object
        % @param double[][] imageX, imageY, image space coordinates
        % @param double[][] psfscX,psfscY, point spread function
        % coordinates
        % @param double[][] P, aperture function
        function out = getPSFimage(obj,x,y,z,objective,imageX,imageY,xApPlane,yApPlane,psfscX,psfscY,P)
                alpha = -1/(obj.getLambda*objective.getF)*(x+imageX./objective.getMagnification);
                beta = -1/(obj.getLambda*objective.getF)*(y+imageY./objective.getMagnification);
                k = 2*pi/obj.getLambda; %wavevector in mm^-1
                Q = P.*exp(-1i*k/2*(z/objective.getF^2).*(xApPlane.^2+yApPlane.^2));
                c = fftshift(fft2(Q));
                h = c.*conj(c);  
                h_scaled = interp2(psfscX,psfscY,h,alpha,beta,'linear');
                out = 1/objective.getMagnification^2.*h_scaled*1/(objective.getF*obj.getLambda)^4;
                out(isnan(out)) = 0;
        end
        % @param string outputPath, path to save reconstructions
        % @param double mnidx, minimum index of slices to reconstruct
        % @param double mxidx, maximum index of slices to reconstruct
        function reconstruct(obj,mnidx,mxidx,displayBoolean)
            maxMinValues = dlmread(fullfile(obj.getOutputPath,'MaxMinValues.txt'));
            for index = mnidx:mxidx
                sinogram = obj.getSinogram(index);
                slice = iradon(sinogram,obj.getNAngles/obj.getNProj,obj.getInterptype,obj.getFilter,1,size(sinogram,1));
                slice(isnan(slice)) = 0;
                if displayBoolean==1
                    imagesc(slice); axis equal tight; colorbar; drawnow;
                end
                maxMinValues(index,:) = [min(slice(:));max(slice(:))];
                writeSlice = gather(uint16((slice-min(slice(:)))./(max(slice(:))-min(slice(:))).*65535));
                imwrite(writeSlice,strcat(fullfile(obj.getOutputPath,num2str(index,'%05d')),'.tiff'));
            end
            dlmwrite(fullfile(obj.getOutputPath,'MaxMinValues.txt'),maxMinValues,';');
        end
    end
    
end

