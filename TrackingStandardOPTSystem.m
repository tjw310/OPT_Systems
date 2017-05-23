classdef TrackingStandardOPTSystem < Standard4fSystem
    %Adds additional properties such as translation over acquistion cycle
    %in x and z
    
    properties (Access = private)
        motorAxisXMotion %double[] of image space x-displacement from centre of image volume in pixels
        motorAxisZMotion %double[] of image space z-displacement from centre of image volume in pixels
    end
    
    methods
        %constructor
        function obj = TrackingStandardOPTSystem()
            obj = obj@Standard4fSystem();
        end
        %get/set
        function out = getMotorAxisXMotion(obj)
            out = obj.motorAxisXMotion;
        end
        function out = getMotorAxisZMotion(obj)
            out = obj.motorAxisZMotion;
        end
        function setMotorAxisXMotion(obj,xMotion)
            obj.motorAxisXMotion = xMotion;
        end
        function setMotorAxisZMotion(obj,zMotion)
            obj.motorAxisZMotion = zMotion;
        end
    end
    
end

