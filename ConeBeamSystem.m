classdef ConeBeamSystem < OPTSystem
    %Simulation of point objects in cone-beam OPT system
    
    properties (Access = private)
        apertureRadius %size of aperture mm
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
        function out = getApertureRadius(obj)
            out = obj.apertureRadius;
        end
        function out = getApertureDisplacement(obj)
            out = obj.apertureDisplacement;
        end
        function out = getR(obj)
            out = obj.R;
        end
        %@param double R, effective source-detector distance in pixels
        function setR(obj,R)
            obj.R = R;
        end
        %@param double apSz, size of aperture in mm
        function setApertureRadius(obj,apSz)
            obj.apertureRadius = apSz;
        end
        %@param double apDisp, axial aperture displacement away from pupil
        %plane in mm
        function setApertureDisplacement(obj,apDisp)
            obj.apertureDisplacement = apDisp;
        end
        %@param Objective objective, objective class provides focal length
        %and magnfication information
        function calculateApertureDisplacement(obj,objective)
           obj.setApertureDisplacement(objective.getF*objective.getF/(obj.R.*obj.getPixelSize/objective.getMagnification));
        end
            
    end
    
    
    methods
        %@param SimpleConeRecon simpleConeRecon class with nProj, nAngles
        %and other properties
        %@param Objective objective, objective class
        %@param PointObject pointObject, point object class
        function simulate(obj,simpleConeRecon,objective,pointObject)
            if isempty(obj.apertureDisplacement)
                obj.calculateApertureDisplacement(objective);
            end
            obj.apertureDisplacement
            obj.clearProjections;
            obj.setAllProjections(0);
            theta = obj.theta;
            
            for i=1:length(theta)
                t = theta(i);
                objectX = pointObject.getX*cos(t)-pointObject.getZ*sin(t); %object space x loc after rotation
                objectZ = pointObject.getZ*cos(t)+pointObject.getX*sin(t); %object space z loc after rotation
                objectY = pointObject.getY;
                f1 = objective.getF(); %focal length of objective
                nWidth = simpleConeRecon.getWidth(); %width of object space in pixels
                nHeight = simpleConeRecon.getHeight(); %height of object space in pixels
                d = obj.getApertureDisplacement; %axial displacement of aperture in mm
                m = objective.getMagnification; %transverse magnfication of objective
                k = 2*pi/obj.lambda; %wavevector in mm^-1

                xApPlaneMax = 2*objective.getRadiusPP; %max x-coordinate in aperture plane
                yApPlaneMax = nHeight/nWidth*xApPlaneMax; %max y-coordinate in aperture plane
                %[xApPlane,yApPlane] = meshgrid(linspace(-xApPlaneMax,xApPlaneMax,nWidth),linspace(-yApPlaneMax,yApPlaneMax,nHeight));
                [xApPlane,yApPlane] = meshgrid(simpleConeRecon.x./nWidth*2*xApPlaneMax,simpleConeRecon.y./nHeight*2*yApPlaneMax);
                rApPlane = sqrt(xApPlane.^2+yApPlane.^2);
                P = ones(size(xApPlane)); P(rApPlane>obj.apertureRadius)=0; %Aperture function (circle)

                xRange = simpleConeRecon.x.*simpleConeRecon.getPixelSize;
                yRange = simpleConeRecon.y.*simpleConeRecon.getPixelSize;
                [imageX,imageY] = meshgrid(xRange,yRange);

                mz = m./(1+objectZ*d/(f1^2)); %Axially dependant magnfication

                Q = P.*exp(-1i*k/2*(objectZ/(f1^2+objectZ*d)).*(xApPlane.^2+yApPlane.^2));

                dx = 2*xApPlaneMax/nWidth;
                uMax = 1/(2*dx);
                
                dy = 2*yApPlaneMax/nHeight;
                vMax = 1/(2*dy);

                c = fftshift(fft2(Q));
                h = c.*conj(c);

                [psfscX,psfscZ] = meshgrid(simpleConeRecon.x./nWidth*2*uMax,simpleConeRecon.y./nHeight*2*vMax);

                alpha = -1/(obj.lambda*f1)*(objectX+imageX./mz);
                beta = -1/(obj.lambda*f1)*(objectY+imageY./mz);
                h_scaled = interp2(psfscX,psfscZ,h,alpha,beta,'linear');

                I = 1/m^2*(mz/m).^2.*h_scaled*1/(f1*obj.lambda)^4;
                I(isnan(I)) = 0;
                
                simpleConeRecon.setProjection(I,i);
                
                imagesc(I); axis equal tight; drawnow;
            end
                
        end
        
    end
    
end

