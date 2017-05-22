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
            effNA = apertureRadius/obj.radiusPP*obj.NA;
        end
        
        %gets effective object space FOV from image field FOV and mag
        function FOV = getFOV(obj)
            FOV = 26.5/obj.magnification;
        end
        
        %gets approximate depth of field using full NA
        % @param pxSz - pixel size of detector in mm
        % @param n - refractive index of immersion medium
        % @param lambda - wavelength of light (mm)
        function DoF = getDoF(obj,pxSz,lambda)
            DoF = n*(lambda/(obj.NA)^2+pxSz/(obj.magnification*obj.NA));      
        end
        
        %gets approximate depth of field using effective NA
        % @param pxSz - pixel size of detector in mm
        % @param n - refractive index of immersion medium
        % @param lambda - wavelength of light (mm)
        % @param apertureRadius - radius of aperture placed in pupil plane
        function DoF = getEffDoF(obj,pxSz,lambda,apertureRadius)
            reducedNA = obj.getEffNA(apertureRadius);
            DoF = n*(lambda/(reducedNA)^2+pxSz/(obj.magnification*reducedNA));      
        end
    end
            
end

