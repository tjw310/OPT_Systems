%script to set up cone beam system with simulation 19th May 2017. T.Watson

NA = 0.3; mag = 10; apertureReductionFactor = 10; lambda = 500e-6;
objective = Objective(NA,mag);
object = PointObject(0,0,0);
stepperMotor = StepperMotor();
%system = Standard4fSystem();
%coneSystem = ConeBeamSystem();

%%
system.setLambda(lambda);
system.setWidth(2048);
system.setHeight(2048);
system.setAxisDirection('vert');
system.setRotationDirection('clock');
system.setNAngles(360);
system.setNProj(100);
system.setBinFactor(4);
system.setApertureRadius(objective.getRadiusPP/apertureReductionFactor);
system.setOpticCentre([0,0]);

%%
coneSystem.setR(10e3);

coneSystem.setBinFactor(system.getBinFactor);
coneSystem.setLambda(system.getLambda);
coneSystem.setWidth(system.getWidth*system.getBinFactor);
coneSystem.setHeight(system.getHeight*system.getBinFactor);
coneSystem.setAxisDirection(system.getAxisDirection);
coneSystem.setRotationDirection(system.getRotationDirection);
coneSystem.setNAngles(system.getNAngles);
coneSystem.setNProj(system.getNProj);
coneSystem.setApertureRadius(system.getApertureRadius);
coneSystem.setOpticCentre(system.getOpticCentre*system.getBinFactor);


