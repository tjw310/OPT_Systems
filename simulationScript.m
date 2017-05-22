%script to set up cone beam system with simulation 19th May 2017. T.Watson

%system = SimpleConeRecon();
NA = 0.4; mag = 20; axialDisplacement = -150; apertureReductionFactor = 10; lambda = 500e-6;
objective = Objective(0.4,-20);
object = PointObject(0.1,-0.25,0);
coneSystem = ConeBeamSystem(objective.getRadiusPP/apertureReductionFactor,lambda);

system.setWidth(1048);
system.setHeight(2110);
system.setAxisDirection('vert');
system.setRotationDirection('anti');
system.setNAngles(360);
system.setNProj(128);
system.setR(10000);
system.setBinFactor(2);


system.simulateProjections(objective,object,coneSystem);

%system.reconPS(object.getY,object.getY)

