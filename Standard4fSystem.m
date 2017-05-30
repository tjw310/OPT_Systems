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
        % @param boolean varargin, display bool, true==new figure and
        % display psf
        function out = getPSFimage(obj,x,y,z,objective,imageX,imageY,xApPlane,yApPlane,psfscX,psfscY,P,~,varargin)
                alpha = -1/(obj.getLambda*objective.getF)*(x+imageX./objective.getMagnification);
                beta = -1/(obj.getLambda*objective.getF)*(y+imageY./objective.getMagnification);
                k = 2*pi/obj.getLambda; %wavevector in mm^-1
                Q = P.*exp(-1i*k/2*(z/objective.getF^2).*(xApPlane.^2+yApPlane.^2));
                c = fftshift(fft2(Q));
                h = c.*conj(c);
                h_scaled = double(interp2(psfscX,psfscY,h,alpha,beta,'linear'));
                out = h_scaled./nansum(h_scaled(:)).*pi*objective.getEffNA(obj.getApertureRadius)^2;
                %out = double(1/objective.getMagnification^2.*h_scaled*1/(objective.getF*obj.getLambda)^4);
                out(isnan(out)) = 0;
                
                if nargin==14 && varargin{1}
                figure; imagesc(imageX(1,:)/objective.getMagnification,imageY(:,1)/objective.getMagnification,out); axis square; drawnow;
            end
        end
        
        % @param string outputPath, path to save reconstructions
        % @param double mnidx, minimum index of slices to reconstruct
        % @param double mxidx, maximum index of slices to reconstruct
        % @param StepperMotor stepperMotor, provides AoR displacement
        % information
        % @param Objective objective, provides magnfication information
        function reconstruct(obj,mnidx,mxidx,stepperMotor,objective,displayBoolean)
            maxMinValues = dlmread(fullfile(obj.getOutputPath,'MaxMinValues.txt'));
            [xShift,zShift] = meshgrid((1:obj.getWidth)+stepperMotor.getX/obj.getPixelSize*objective.getMagnification,(1:obj.getWidth)-stepperMotor.getZ/obj.getPixelSize*objective.getMagnification);
            for index = mnidx:mxidx
                sinogram = obj.getShiftedSinogram(index,stepperMotor,objective);
                slice = iradon(circshift(sinogram,[-1,0]),obj.getNAngles/obj.getNProj,obj.getInterptype,obj.getFilter,1,size(sinogram,1));
                slice = interp2(slice,xShift,zShift);
                slice(isnan(slice)) = 0;
                if displayBoolean==1
                    imagesc(slice); axis equal tight; colorbar; drawnow;
                end
                maxMinValues(index,1:3) = [min(slice(:)),max(slice(:)),index];
                writeSlice = gather(uint16((slice-min(slice(:)))./(max(slice(:))-min(slice(:))).*65535));
                imwrite(writeSlice,strcat(fullfile(obj.getOutputPath,num2str(index,'%05d')),'.tiff'));
            end
            dlmwrite(fullfile(obj.getOutputPath,'MaxMinValues.txt'),maxMinValues,';');
        end
    
    end
    
end

