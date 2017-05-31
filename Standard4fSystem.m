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

