classdef ConeBeamSystem < OPTSystem
    %Simulation of point objects in cone-beam OPT system
    
    properties (Access = private)
        apertureDisplacement %axial displacement of aperture from pupil plane mm
        R %effective source-detector distance (pixels) (unbinned raw value)
    end
    
    methods %constructor
        function obj = ConeBeamSystem()
            obj = obj@OPTSystem();
        end
    end
    
    %% get/set methods
    methods      
        function out = getApertureDisplacement(obj)
            out = obj.apertureDisplacement;
        end
        function out = getR(obj)
            out = obj.R/obj.getBinFactor;
        end
        %@param double R, effective source-detector distance in pixels
        function setR(obj,R)
            obj.R = R;
        end
        %@param double apDisp, axial aperture displacement away from pupil
        %plane in mm
        function setApertureDisplacement(obj,apDisp)
            obj.apertureDisplacement = apDisp;
        end 
        %@param Objective objective, objective class provides focal length
        %and magnfication information
        function calculateApertureDisplacement(obj,objective)
           obj.setApertureDisplacement(objective.getF*objective.getF/(obj.getR.*obj.getPixelSize/objective.getMagnification));
        end         
    end

    methods  
        % @param double x,y,z , location of point object
        % @param Objective objective, objective class object
        % @param double[][] imageX, imageY, image space coordinates
        % @param double[][] psfscX,psfscY, point spread function
        % coordinates
        % @param double[][] P, aperture function
        % @param double[] opticCentre, binned opticCentre [x,y] in mm from
        % centre of FOV
        function out = getPSFimage(obj,x,y,z,objective,imageX,imageY,xApPlane,yApPlane,psfscX,psfscY,P,opticCentre)
            obj.calculateApertureDisplacement(objective);
            mz = objective.getMagnification./(1+z*obj.apertureDisplacement/(objective.getF^2)); %Axially dependant magnfication
            alpha = -1/(obj.getLambda*objective.getF)*(x+(imageX-opticCentre(1))./mz+opticCentre(1)/objective.getMagnification);
            beta = -1/(obj.getLambda*objective.getF)*(y+(imageY-opticCentre(2))./mz+opticCentre(2)/objective.getMagnification);
            k = 2*pi/obj.getLambda; %wavevector in mm^-1
            Q = P.*exp(-1i*k/2*(z/(objective.getF^2+z*obj.apertureDisplacement)).*(xApPlane.^2+yApPlane.^2));
            c = fftshift(fft2(Q));
            h = c.*conj(c);  
            h_scaled = interp2(psfscX,psfscY,h,alpha,beta,'linear');
            %out = double(1/objective.getMagnification^4.*mz^2.*h_scaled*1/(objective.getF*obj.getLambda)^4);
            out = h_scaled./nansum(h_scaled(:)).*(mz/objective.getMagnification)^2;
            out(isnan(out)) = 0; 
        end
        
        % @param string outputPath, path to save reconstructions
        % @param double mnidx, minimum index of slices to reconstruct
        % @param double mxidx, maximum index of slices to reconstruct
        % @param boolean displayBoolean, true if want to display
        % @param StepperMotor stepperMotor, class with rotation axis prop
        function reconstruct(obj,mnidx,mxidx,stepperMotor,objective,displayBoolean)
            if isempty(obj.getAllFilteredProj)
                obj.filterProjections();
            end
            %common parameters to send to reconstruction function
            [xx,zz] = meshgrid(obj.xPixels,obj.zPixels);
            t = 2*pi-obj.theta;
            if obj.getUseGPU == 1
                xxS = gpuArray(xx+stepperMotor.getX/obj.getPixelSize*objective.getMagnification);
                zz = gpuArray(zz);
            else
                xxS = xx+stepperMotor.getX/obj.getPixelSize*objective.getMagnification;
            end
            maxMinValues = dlmread(fullfile(obj.getOutputPath,'MaxMinValues.txt'));
            for index=mnidx:mxidx
                op = obj.getOpticCentre;        
                D = obj.getR;
                for i = 1:obj.getNProj
                    motorOffset = -stepperMotor.currentX(i)/obj.getPixelSize*objective.getMagnification;
                    if obj.getUseGPU==1
                        projI = gpuArray(obj.getFilteredProj(i));
                        if i==1
                            slice = gpuArray(zeros(obj.getWidth,obj.getWidth));
                        end
                    else
                        if i==1
                            slice = zeros(obj.getWidth,obj.getWidth);
                        end
                        projI = (obj.getFilteredProj(i));
                    end

                   u = D.*(xxS*cos(t(i))+zz*sin(t(i))-op(1)+motorOffset)...
                       ./(xxS*sin(t(i))-zz*cos(t(i))+D)+op(1)-motorOffset;

                   U = (xxS*sin(t(i))-zz*cos(t(i))+D)./D;
                   v = (1./U.*(index-obj.getHeight/2-op(2)))+op(2);                       

                    u2 = u-xxS(1,1)+1;
                    v2 = v-zz(1,1)+1;

                    switch obj.getAxisDirection
                        case 'horiz'
                            z = interp2(projI,v2,u2,obj.getInterptype);
                        case 'vert'
                            z = interp2(projI,u2,v2,obj.getInterptype);
                    end
                    plane = z.*D./U.^2;

                    plane(isnan(plane))=0;
                    slice = slice + plane;
                    if rem(i,round(obj.getNProj/20))==0
                        disp(sprintf('Reconstruction Completion Percentage: %.1f%%',(obj.getNProj*(index-mnidx)+i)/(obj.getNProj*(mxidx-mnidx+1))*100));
                    end
                end
                slice(isnan(slice)) = 0;
                maxMinValues(index,1:3) = [min(slice(:)),max(slice(:)),index];
                writeSlice = gather(uint16((slice-min(slice(:)))./(max(slice(:))-min(slice(:))).*65535));
                imwrite(writeSlice,strcat(fullfile(obj.getOutputPath,num2str(index,'%05d')),'.tiff'));
                if displayBoolean==1
                    imagesc(slice); axis equal tight; colorbar(); drawnow;
                end
            end
            dlmwrite(fullfile(obj.getOutputPath,'MaxMinValues.txt'),maxMinValues,';');
        end
        
    end
    
end

