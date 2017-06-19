classdef Objective < handle
    %Class representing the objective used in an Olympus IX71 System.
    %Used with a f=180mm tube lens. All length measurements in mm. Output
    %image field diameter = 26.5mm.
    
    %% Properties
    properties (Access = private)
        NA
        workingDistance   
        magnification
        radiusPP %radius of the pupil plane
    end
    
    %% constructors
    methods 
        function obj = Objective(NA,magnification,varargin)
            if nargin > 3
                error('Too many arguments');
            else
                obj.NA = NA;
                obj.magnification=magnification;
                if nargin==3
                    obj.workingDistance = varargin{1};
                else
                    obj.workingDistance = [];
                end
            end
            obj.radiusPP = obj.getF*obj.NA;
        end
    end
    
    %% get set methods
    methods
        function out = getRadiusPP(obj)
            out = obj.radiusPP;
        end
        function out = getNA(obj)
            out = obj.NA;
        end
        function out = getMagnification(obj)
            out = obj.magnification;
        end
        function out = getWD(obj)
            out = obj.workingDistance;
        end
        %get max spatial frequnecy transfer (Abbe criterion) for incoherent
        %detection
        % @param OPTSystem optSys, sends lambda parameter
        function out = getAbbe(obj,optSys)
            out = 2*obj.getEffNA(optSys.getApertureRadius)/optSys.getLambda;
        end
        %@param double NA, numerical aperture
        function setNA(obj,NA)
            obj.NA = NA;
        end
        %@param double workingDistance, cannot be <0
        function setWD(obj,workingDistance)
            obj.workingDistance = workingDistance;
        end
        %@param double magnification, sets objective magnification
        function setMagnification(obj,magnification)
            obj.magnification = magnification;
        end
    end
    
    %% Common parameters access
    methods
        %get focal length of objective
        function f = getF(obj)
            f = 180/obj.magnification;
        end
        
        %get effective NA using an aperture of radius apertureRadius (mm)
        function effNA = getEffNA(obj,apertureRadius)
            if apertureRadius<obj.radiusPP
                effNA = apertureRadius/obj.radiusPP*obj.NA;
            else
                effNA = obj.NA;
            end
        end
        
        %gets effective object space FOV from image field FOV and mag
        function FOV = getFOV(obj)
            FOV = 26.5/obj.magnification;
        end
        
        %gets approximate depth of field using full NA
        % @param pxSz - pixel size of detector in mm
        % @param n - refractive index of immersion medium
        % @param lambda - wavelength of light (mm)
        function DoF = getDoF(obj,pxSz,lambda,n)
            DoF = n*(lambda/(obj.NA)^2+pxSz/(obj.magnification*obj.NA));      
        end
        
        %gets traditional depth of field limit: DoF = n*4*lambda/(NA^2)
        % @param n - refractive index of immersion medium
        % @param lambda - wavelength of light (mm)
        function DoF = getTraditionalDoF(obj,lambda,n)
            DoF = n*4*lambda/(obj.NA^2);
        end
        
        %gets approximate depth of field using effective NA
        % @param pxSz - pixel size of detector in mm
        % @param n - refractive index of immersion medium
        % @param lambda - wavelength of light (mm)
        % @param apertureRadius - radius of aperture placed in pupil plane
        function DoF = getEffDoF(obj,pxSz,lambda,apertureRadius,n)            
            reducedNA = obj.getEffNA(apertureRadius);
            DoF = n*(lambda/(reducedNA)^2+pxSz/(obj.magnification*reducedNA));      
        end
        
        %gets traditional effective depth of field limit: DoF =
        %n*4*lambda/NA^2 for OPT system with aperture
        % @param n - refractive index of immersion medium
        % @param lambda - wavelength of light (mm)
        % @param apertureRadius - radius of aperture placed in pupil plane
        function DoF = getEffTradDoF(obj,lambda,apertureRadius,n)
            DoF = n*4*lambda/(obj.getEffNA(apertureRadius)^2);
        end
        
        %gets maximum focal displacement when using a tunable lens in a
        %relay system
        % @param double maxPower, max ETL optical power = 1/f (f in mm)
        % @oaram double minPower, min ETL optical power
        % @param double relayLensFocalLength, focal length of primary relay
        % lens in mm
        function out = MFD(obj,maxPower,minPower,relayLensFocalLength)
            out = abs((relayLensFocalLength/obj.magnification)^2*(maxPower-minPower));
        end
        
        %gets maximum focal displacement when using a tunable lens in a
        %non telecentric (cone) OPT system
        % @param double maxPower, max ETL optical power = 1/f (f in mm)
        % @oaram double minPower, min ETL optical power
        % @param double apertureDisplacement
        function out = coneMFD(obj,maxPower,minPower,apertureDisplacement)
            d = apertureDisplacement;
            out = abs(obj.getF^2*(maxPower-minPower)./((1+d*maxPower)*(1+d*minPower)));
        end
        
        %plots MFD ratio for cone vs standard OPT systems
        % @param double maxPower, max ETL optical power = 1/f (f in mm)
        % @oaram double minPower, min ETL optical power
        % @param double relayLensFocalLength, focal length of primary relay
        % lens in mm
        % @param double radiusETL, radius of ETL
        % @param double[] apertureDisplacementRange, range of aperture
        % displacements to plot
        % @return double[] coneMFD
        % @return double perfectStandardMFD
        % @return double realisticStandardMFD
        function [coneMFD,perfectStandardMFD,realisticStandardMFD] = plotMFDRatio(obj,maxPower,minPower,relayLensFocalLength,radiusETL,apertureDisplacementRange,varargin)
            perfectStandardSystemFocalLength = radiusETL/obj.radiusPP*180;
            perfectStandardMFD = obj.MFD(maxPower,minPower,perfectStandardSystemFocalLength);
            realisticStandardMFD = obj.MFD(maxPower,minPower,relayLensFocalLength);
            coneMFD = abs(obj.getF^2*(maxPower-minPower)./((1+apertureDisplacementRange*maxPower).*(1+apertureDisplacementRange*minPower)));
            figure;
            subplot(1,2,1); hold on; plot(apertureDisplacementRange,(coneMFD./perfectStandardMFD)); hold off; axis square;
            subplot(1,2,2); hold on; plot(apertureDisplacementRange,(coneMFD./realisticStandardMFD)); hold off;axis square;

        end
    end
            
end

