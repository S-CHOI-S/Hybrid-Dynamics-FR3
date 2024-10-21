

#include "controller.h"
#include <chrono>
#include <vector>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fstream> // ifstream header
#include <iostream>
#include <string> // getline header
#include <math.h>
#include <time.h>
// #include <torch/extension.h>
// #include <torch/torch.h>
// #include <torch/script.h>

const string RED = "\033[31m";
const string GREEN = "\033[32m";
const string YELLOW = "\033[33m";
const string BLUE = "\033[34m";
const string MAGENTA = "\033[35m";
const string CYAN = "\033[36m";
const string RESET = "\033[0m";

CController::CController(int JDOF)
{
	_k = JDOF;
	// Initialize();
}

CController::~CController()
{
}

// for pybind11
////////////////////////////////////////////////////////////////////////////////////////////////
void CController::read_pybind(double t, array<double, 9> q, array<double, 9> qdot, double timestep)
{
	_dt = timestep;
	_t = t;
	if (_bool_init == true)
	{
		_init_t = _t;
		for (int i = 0; i < 1; i++)
		{
			_init_gripper = q[_k + i];
		}

		_bool_init = false;
	}

	// _dt = t - _pre_t;
	// _pre_t = t;

	for (int i = 0; i < _k; i++)
	{
		_q(i) = q[i];
		_qdot(i) = qdot[i];
	}
	for (int i = 0; i < 1; i++)
	{
		_gripper = q[_k + i];
		_gripperdot = qdot[_k + i];
	}
}

tuple<std::vector<double>, double> CController::write_pybind()
{
	torque_command.clear();

	for (int i = 0; i < _k; i++)
	{
		torque_command.push_back(_torque(i));
		// torque_command.push_back(Model._bg(i));
	}
	for (int i = 0; i < 1; i++)
	{
		torque_command.push_back(_grippertorque);
	}
	if (_control_mode == 4)
	{
		return make_tuple(torque_command, abs(_theta_des - _init_theta));
	}
	else
	{
		return make_tuple(torque_command, 0);
	}
}

// double CController::write_force_pybind()
// {
// 	return _force_gain;
// }

void CController::put_action_pybind(array<double, 7> torque)
{
	for (int i=0; i<7; i++)
	{
		action_torque[i] = torque[i];
	}
}

double CController::lpf(double input, double previousOutput, double alpha) {
    return alpha * input + (1.0 - alpha) * previousOutput;
}

// void CController::randomize_env_pybind(std::array<std::array<double, 3>, 3> rotation_obj, std::string object_name, int scale, std::array<double, 66> pos, double init_theta, double goal_theta, int planning_mode, bool generate_dxyz)
// {
// 	_init_theta = init_theta;
// 	_goal_theta = goal_theta;
// 	_object_name = object_name;
// 	_planning_mode = planning_mode;
// 	_generate_dxyz = generate_dxyz;
// 	_scale_obj = scale;
// 	int robot_base = 6;
// 	int valve = 42;
// 	int handle_valve = 54;
// 	for (int i = 0; i < 3; ++i)
// 	{
// 		for (int j = 0; j < 3; ++j)
// 		{
// 			_rotation_obj(i, j) = rotation_obj[i][j];
// 		}
// 	}
// 	for (int i = 0; i < 3; i++)
// 	{
// 		_valve(i) = pos[valve + i];
// 		_handle_valve(i) = pos[handle_valve + i];
// 		_robot_base(i) = pos[robot_base + i];
// 	}
// 	// _handle_valve(1) += 0.02;
// }

array<double, 6> CController::get_commands_pybind(){
	array<double, 6> commands = {_droll, _dpitch, _roll, _pitch, _force_gain, _rforce_gain};
	return commands;
}

tuple<std::vector<double>, std::vector<double>> CController::get_force_pybind()
{
	torque_command.clear();
	force_command.clear();

	for (int i = 0; i < _k; i++)
	{
		torque_command.push_back(_torque(i));
	}
	for (int i = 0; i < 1; i++)
	{
		torque_command.push_back(_grippertorque);
	}
	for (int i = 0; i < 6; i++)
	{
		force_command.push_back(_compensated_force(i));
	}
	return make_tuple(force_command, torque_command);
}

double CController::control_mode_pybind()
{
	return _control_mode;
}

double CController::write_control_mode_pybind(double ctrl_mode)
{
	_control_mode = ctrl_mode;
}

vector<vector<double>> CController::get_jacobian_pybind()
{
	J_hands.clear();
	J_hands = {{{0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0}}};
	for (int i = 0; i < _J_hands.rows(); ++i)
	{
		for (int j = 0; j < _J_hands.cols(); ++j)
		{
			J_hands[i][j] = _J_hands(i, j);
		}
	}

	return J_hands;
}

tuple<vector<vector<double>>, vector<vector<double>>, vector<vector<double>>> CController::get_model_pybind()
{
	J_hands.clear();
	J_hands = {{{0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0}}};
	for (int i = 0; i < _J_hands.rows(); ++i)
	{
		for (int j = 0; j < _J_hands.cols(); ++j)
		{
			J_hands[i][j] = _J_hands(i, j);
		}
	}
	inertia.clear();
	inertia = {{{0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0}}};
	for (int i = 0; i < Model._A.rows(); ++i)
	{
		for (int j = 0; j < Model._A.cols(); ++j)
		{
			inertia[i][j] = Model._A(i, j);
		}
	}
	lambda.clear();
	lambda = {{{0, 0, 0, 0, 0, 0},
			   {0, 0, 0, 0, 0, 0},
			   {0, 0, 0, 0, 0, 0},
			   {0, 0, 0, 0, 0, 0},
			   {0, 0, 0, 0, 0, 0},
			   {0, 0, 0, 0, 0, 0}}};
	for (int i = 0; i < _lambda.rows(); ++i)
	{
		for (int j = 0; j < _lambda.cols(); ++j)
		{
			lambda[i][j] = _lambda(i, j);
		}
	}

	return make_tuple(J_hands, inertia, lambda);
}

tuple<vector<double>, vector<double>, float, float, float, float> CController::get_ee_pybind()
{
	x_hand.clear();
	xdot_hand.clear();

	for (int i = 0; i < 6; i++)
	{
		xdot_hand.push_back(_xdot_des_hand(i));
	}
	for (int i = 0; i < 6; i++)
	{
		x_hand.push_back(_x_hand(i));
	}
	return make_tuple(xdot_hand, x_hand, _roll, _pitch, _droll, _dpitch);
}

vector<double> CController::desired_rpy_pybind()
{
	rpy_des.clear();
	_rpy_des = CircularTrajectory.rotation_circular(_pitch, _roll, _x_hand.head(3));
	for (int i = 0; i < 3; i++)
	{
		rpy_des.push_back(_rpy_des(i));
	}

	return rpy_des;
}

tuple<array<double, 2>, double> CController::get_actions_pybind()
{
	array<double, 2> action_rotation;
	double action_force = _dforce_gain;
	action_rotation = {_ddroll, _ddpitch};

	return make_tuple(action_rotation, action_force);
}
// tuple<array<double, 2>, double> CController::get_commands_pybind()
// {
// 	array<double, 2> action_rotation;
// 	double action_force = _force_gain;
// 	action_rotation = {_roll, _pitch};

// 	return make_tuple(action_rotation, action_force);
// }

array<double, 16> CController::relative_T_hand_pybind()
{
	MatrixXd T_hand(4, 4);
	T_hand.block<3, 3>(0, 0) << _R_hand;
	T_hand.block<3, 1>(0, 3) << _x_hand.head(3);
	T_hand.block<1, 4>(3, 0) << 0, 0, 0, 1;

	MatrixXd T_v_hand = _Tvr.inverse() * T_hand;
	array<double, 16> arr;
	for (int i = 0; i < T_v_hand.rows(); ++i)
	{
		for (int j = 0; j < T_v_hand.cols(); ++j)
		{
			arr[i * T_v_hand.rows() + j] = T_v_hand(i, j);
		}
	}

	return arr;
}

////////////////////////////////////////////////////////////////////////////////////////////////

void CController::control_mujoco()
{
	ModelUpdate();
	motionPlan();
	if (_control_mode == 1) // joint space control
	{
		if (_t - _init_t < 0.1 && _bool_joint_motion == false)
		{
			VectorXd tmp;
			tmp.setZero(7);

			_start_time = _init_t;
			_end_time = _start_time + _motion_time;
			JointTrajectory.reset_initial(_start_time, _q, tmp);
			JointTrajectory.update_goal(_q_goal, _qdot_goal, _end_time);
			_bool_joint_motion = true;
			_x_des_hand = _x_hand;
			_xdot_des_hand = _xdot_hand;
			_q_des = _q;
			_qdot_des = _qdot;
		}

		JointTrajectory.update_time(_t);
		_q_des = JointTrajectory.position_cubicSpline();
		_qdot_des = JointTrajectory.velocity_cubicSpline();

		JointControl();
		GripperControl(); // planning + torque generation
		if (JointTrajectory.check_trajectory_complete() == 1)
		{
			_bool_plan(_cnt_plan) = 1;
			_bool_init = true;
		}
	}
	else if (_control_mode == 2) // task space control
	{
		if (_t - _init_t < 0.1 && _bool_ee_motion == false)
		{
			_start_time = _init_t;
			_end_time = _start_time + _motion_time;
			HandTrajectory.reset_initial(_start_time, _x_hand, _xdot_hand);
			HandTrajectory.update_goal(_x_goal_hand, _xdot_goal_hand, _end_time);
			_bool_ee_motion = true;

			_x_des_hand = _x_hand;
			_xdot_des_hand = _xdot_hand;
			_q_des = _q;
			_qdot_des = _qdot;
			_theta_des = 0.0;
		}
		
		HandTrajectory.update_time(_t);
		_x_des_hand.head(3) = HandTrajectory.position_cubicSpline();
		_xdot_des_hand.head(3) = HandTrajectory.velocity_cubicSpline();
		_R_des_hand = HandTrajectory.rotationCubic();
		_x_des_hand.segment<3>(3) = CustomMath::GetBodyRotationAngle(_R_des_hand);
		_xdot_des_hand.segment<3>(3) = HandTrajectory.rotationCubicDot();
		
		// CLIK();
		// OperationalSpaceControl();
		HybridControl();
		GripperControl();

		// _q_plot.row(_cnt_plot) = _q;
		// _qdot_plot.row(_cnt_plot) = _qdot;
		// _x_plot.row(_cnt_plot) = _x_hand;
		// _cnt_plot += 1;

		if (HandTrajectory.check_trajectory_complete() == 1)
		{
			_bool_plan(_cnt_plan) = 1;
			_bool_init = true;
		}
	}
	else if (_control_mode == 3) // task space control
	{
		if (_t - _init_t < 0.1 && _bool_ee_motion == false)
		{
					
			_start_time = _init_t;
			_end_time = _start_time + _motion_time;
			HandTrajectory.reset_initial(_start_time, _x_hand, _xdot_hand);
			HandTrajectory.update_goal(_x_goal_hand, _xdot_goal_hand, _end_time);
			_bool_ee_motion = true;

			_x_des_hand = _x_hand;
			_xdot_des_hand = _xdot_hand;
			_q_des = _q;
			_qdot_des = _qdot;
			_theta_des = 0.0;
			_R_hand_ref = _R_hand;
		}
		
		HandTrajectory.update_time(_t);
		_x_des_hand.head(3) = HandTrajectory.position_cubicSpline();
		_xdot_des_hand.head(3) = HandTrajectory.velocity_cubicSpline();
		
		_R_des_hand = rotation_calc(_roll, _pitch, _R_hand_ref);
		_x_des_hand.segment<3>(3) = CustomMath::GetBodyRotationAngle(_R_des_hand);
		_xdot_des_hand.segment<3>(3) = rotationdot_circular(_R_des_hand, _R_hand);
		_force_gain += _dforce_gain*0.1;

		if (_force_gain < 0.0)
		{
			_force_gain = 0.0;
		}

		// CLIK();
		// OperationalSpaceControl();
		// _force_gain += 0.01;
		// if (_force_gain > 10){
		// 	_force_gain = 10;
		// }

		// Manual
		// _roll = 0.0;
		// _pitch = 0.0;

		// if (_qdot_cabinet < abs(0.2) * 1)
		// {
		// 	_dforce_gain = 0.1;
		// 	_force_gain += _dforce_gain * 0.1;
		// 	// cout << "AAA" << endl;
		// }
		// else if (_qdot_cabinet > abs(0.3) * 1)
		// {
		// 	_dforce_gain = -0.1;
		// 	_force_gain += _dforce_gain * 0.1;
		// 	// cout << "BBB" << endl;
		// }
		// else
		// {
		// 	_dforce_gain = 0.0;
		// 	_force_gain += _dforce_gain * 0.1;
		// 	// cout << "CCC" << endl;
		// }
		
		// VectorXd min_q(7), max_q(7), scaled_q(7);
		// min_q << -2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159;
		// max_q << 2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159;
		// scaled_q << ((_q - min_q).cwiseQuotient(max_q - min_q) * (1 - (-1))).array() - 1;
		// if (scaled_q.cwiseAbs().maxCoeff() > 0.9)
		// {
		// 	_dforce_gain = -100;
		// 	_force_gain += _dforce_gain * 0.1;
		// }

		// if (_force_gain < 0.0)
		// {
		// 	_force_gain = 0.0;
		// }

		HybridControl();
		GripperControl();

		// _q_plot.row(_cnt_plot) = _q;
		// _qdot_plot.row(_cnt_plot) = _qdot;
		// _x_plot.row(_cnt_plot) = _x_hand;
		// _cnt_plot += 1;

		if (HandTrajectory.check_trajectory_complete() == 1)
		{
			_bool_plan(_cnt_plan) = 1;
			_bool_init = true;

		}
	}
	else if (_control_mode == 5) // torque control
	{
		if (_t - _init_t < 0.1)
		{
					
			_start_time = _init_t;
			_end_time = _start_time + _motion_time;
		}

		_torque = action_torque;
	}

	
	_q_pre = _q;
	_qdot_pre = _qdot;
	// cout<< "====================================" << endl;
	// cout<< _qdot << endl;
}

double CController::get_force_gain()
{
	// cout<< "force_gain: " << _force_gain << endl;
	return _dforce_gain;
}

void CController::ModelUpdate()
{
	Model.update_kinematics(_q, _qdot);
	Model.update_dynamics();

	Model.calculate_EE_Jacobians();

	Model.calculate_EE_positions_orientations();

	Model.calculate_EE_velocity();

	_J_hands = Model._J_hand;
	_x_hand.head(3) = Model._x_hand;
	// Matrix3d EE_align1;
	// EE_align1 << cos(-M_PI_4), sin(-M_PI_4), 0, -sin(-M_PI_4), cos(-M_PI_4), 0, 0, 0, 1;
	// _R_hand = EE_align1 * Model._R_hand;
	_R_hand = Model._R_hand;
	_x_hand.tail(3) = CustomMath::GetBodyRotationAngle(_R_hand);
	// Matrix3d EE_align1, EE_align2, EE_align3, EE_align4 ;
	// EE_align1<< cos(-M_PI_4), sin(-M_PI_4),0,-sin(-M_PI_4),cos(-M_PI_4),0, 0,0,1;
	// EE_align2<< cos(-M_PI_4), -sin(-M_PI_4),0,sin(-M_PI_4),cos(-M_PI_4),0, 0,0,1;
	// EE_align3<< cos(M_PI_4), sin(M_PI_4),0,-sin(M_PI_4),cos(M_PI_4),0, 0,0,1;
	// EE_align4<< cos(M_PI_4), -sin(M_PI_4),0,sin(M_PI_4),cos(M_PI_4),0, 0,0,1;
	// 	// CustomMath::GetBodyRotationAngle(Model._R_hand)
	// cout<<"current hand1 : "<<CustomMath::GetBodyRotationAngle(EE_align1*_R_hand).transpose()<<endl;
	// cout<<"current hand2 : "<<CustomMath::GetBodyRotationAngle(EE_align2*_R_hand).transpose()<<endl;
	// cout<<"current hand3 : "<<CustomMath::GetBodyRotationAngle(EE_align3*_R_hand).transpose()<<endl;
	// cout<<"current hand4 : "<<CustomMath::GetBodyRotationAngle(EE_align4*_R_hand).transpose()<<endl;

	_xdot_hand = Model._xdot_hand;
	// for (int i = 0; i < 7; i++)
	// {
	// 	Model._A(i, i) += 0.1;
	// }
}

Vector3d CController::AddTorque()
{
	Vector4d tangential_vector;
	tangential_vector << -_obj.o_margin.normalized(), 0;
	if (_init_theta < _goal_theta)
	{
		// 반시계방향
		tangential_vector << -_obj.o_margin.normalized().cross((_obj.pos + _obj.o_margin - _x_hand.head(3)).normalized()), 0;
	}
	else
	{
		tangential_vector << _obj.o_margin.normalized().cross((_obj.pos + _obj.o_margin - _x_hand.head(3)).normalized()), 0;
	}
	tangential_vector = _Tur * tangential_vector; // robot

	return tangential_vector.head(3);
}

void CController::load_model()
{

	//   cout << "PyTorch version: "
	// 	<< TORCH_VERSION_MAJOR << "."
	// 	<< TORCH_VERSION_MINOR << "."
	// 	<< TORCH_VERSION_PATCH << std::endl;

	// torch::jit::script::Module module;
	// try {
	//     // Load the TorchScript model
	//     module = torch::jit::load("/home/kist-robot2/catkin_ws/src/franka_emika_panda/py_src/RL_actor.pt");
	// } catch (const c10::Error& e) {
	//     std::cerr << "Error loading the model: " << e.msg() << std::endl;

	// }
	// torch::Tensor input_tensor = torch::ones({1, 139});
	// at::Tensor output = module.forward({input_tensor}).toTensor();
	// std::cout << "Output tensor: " << output << std::endl;
}

void CController::writeToCSVfile(string filename, MatrixXd input_matrix, int cnt)
{
	// Open the file for writing
	string path = "/home/kist-robot2/catkin_ws/src/franka_overall/py_src/data/compare_dynamics/";
	path = path + filename;
	std::ofstream outputFile(path);

	if (!outputFile.is_open())
	{
		std::cerr << "Error opening file " << filename << std::endl;
	}
	// outputFile << "qd1,q1,qd2,q2,qd3,q3,qd4,q4,qd5,q5,qd6,q6,qd7,q7,xd,x,yd,y,zd,z,rd,r,pd,p,yd,y"
	//            << endl;
	// Write the matrix data to the CSV file
	int len = input_matrix.cols();
	for (Eigen::Index i = 0; i < cnt; ++i)
	{
		for (Eigen::Index j = 0; j < len; ++j)
		{
			outputFile << input_matrix(i, j);
			outputFile << ",";
		}
		outputFile << "\n"; // Move to the next row
	}
	outputFile.close();
}

Matrix3d CController::R3D(Objects obj, Vector3d unitVec, double angle)
{
	Matrix3d _Tug;
	angle = -angle; // frame은 반대 방향으로 회전 해야지, gripper방향이 유지된다.
	double cosAngle = cos(angle);
	double sinAngle = sin(angle);
	double x = unitVec(0);
	double y = unitVec(1);
	double z = unitVec(2);
	Matrix3d rotMatrix;
	rotMatrix << cosAngle + (1 - cosAngle) * x * x, (1 - cosAngle) * x * y - sinAngle * z, (1 - cosAngle) * x * z + sinAngle * y,
		(1 - cosAngle) * y * x + sinAngle * z, cosAngle + (1 - cosAngle) * y * y, (1 - cosAngle) * y * z - sinAngle * x,
		(1 - cosAngle) * z * x - sinAngle * y, (1 - cosAngle) * z * y + sinAngle * x, cosAngle + (1 - cosAngle) * z * z;

	_Tug << rotMatrix * obj.grab_dir.normalized(),
		rotMatrix * -obj.o_margin.normalized().cross(obj.grab_dir.normalized()),
		rotMatrix * -obj.o_margin.normalized();
	return _Tug;
}


Matrix3d CController::rotation_calc(double pi, double psi, Matrix3d R_hand){
	
	//_start_time, _grab_vector, _normal_vector, _obj.origin, _radius, _Tvr, _dt
	Matrix3d Tue;
	Matrix3d Troll, Tpitch, Troll_inv, Tpitch_inv;
	Vector3d x_direction, y_direction;
    Vector3d rpy;
    	
	x_direction << _cabinet.r_margin.normalized();
	y_direction << _cabinet.norm_dir.cross(x_direction).normalized();
	Troll = R3D(_cabinet, x_direction, psi);
	Troll_inv =R3D(_cabinet,x_direction, 0);
	Tpitch = R3D(_cabinet,y_direction, pi);
	Tpitch_inv = R3D(_cabinet, y_direction, 0);

	// Tue << Troll*Troll.inverse()*Tpitch*Tpitch_inv.inverse()*Tug*Tge; 
	Tue << Troll*Troll_inv.inverse()*Tpitch*Tpitch_inv.inverse()*R_hand; 
	// rpy = CustomMath::GetBodyRotationAngle(Tue);
	// cout<<"R hand \n"<<R_hand<<"\nR des hand\n"<<Tue<<endl<<endl;
	return Tue;

}
Vector3d CController::rotationdot_circular(Matrix3d r1, Matrix3d r0){
	Matrix3d rdiff = r1 * r0.transpose();
	Matrix3d r_skew = rdiff.log();
	Vector3d drpy;
	drpy(0) = r_skew(2, 1);
	drpy(1) = r_skew(0, 2);
	drpy(2) = r_skew(1, 0);
	
	return drpy/_dt;
}

CController::Target CController::TargetTransformMatrix(Objects obj, Robot robot, double angle)
{
	// frame u : universal
	// frame v : valve rotation axis
	// frame b : valve base orgin (same rotation with frame u)
	// frame g : gripper
	// frame e : end-effector
	Target target;

	// position x,y,z
	Vector4d _xaxis;
	Vector4d _yaxis;
	Vector4d _zaxis;
	Vector4d _porg;
	Vector4d tmp;
	Matrix4d Tvb; // valve handle -> valve base
	Matrix4d Tbu; // valve base -> universal
	Matrix4d Tur; // universal -> robot
	Matrix4d Tvr; // valve handle -> valve vase -> universal -> robot!!

	// roll pitch yaw
	Matrix3d _Tug; // universal -> gripper
	Matrix3d _Tge; // gripper -> end-effector
	Matrix3d _Tue; // universal -> gripper -> end-effector

	// calc target x,y,z
	_xaxis << obj.r_margin.normalized(), 0;
	_yaxis << obj.o_margin.normalized().cross(obj.r_margin.normalized()), 0;
	_zaxis << obj.o_margin.normalized(), 0;
	_porg << obj.o_margin, 1;

	Tvb << _xaxis, _yaxis, _zaxis, _porg;

	Tbu << 1, 0, 0, obj.pos(0),
		0, 1, 0, obj.pos(1),
		0, 0, 1, obj.pos(2),
		0, 0, 0, 1;

	Tur << cos(-robot.zrot), sin(-robot.zrot), 0, robot.pos(0),
		-sin(-robot.zrot), cos(-robot.zrot), 0, robot.pos(1),
		0, 0, 1, -robot.pos(2),
		0, 0, 0, 1;

	Tvr << Tur * Tbu * Tvb;

	tmp << obj.r_margin.norm() * cos(angle), obj.r_margin.norm() * sin(angle), 0, 1;
	tmp << Tvr * tmp;

	target.x = tmp(0);
	target.y = tmp(1);
	target.z = tmp(2);
	_x_plan.push_back(target.x);
	_y_plan.push_back(target.y);
	_z_plan.push_back(target.z);

	_Tge << cos(robot.ee_align), -sin(robot.ee_align), 0,
		sin(robot.ee_align), cos(robot.ee_align), 0,
		0, 0, 1;
	_Tug = CController::R3D(obj, -obj.o_margin.normalized(), angle);

	MatrixXd _Tug_normal, _Tug_y;
	MatrixXd _Tug_normal_inv, _Tug_y_inv;
	_Tug_normal.setZero(4, 4);
	_Tug_normal_inv.setZero(4, 4);
	_Tug_y.setZero(4, 4);
	_Tug_y_inv.setZero(4, 4);

	Vector3d x_direction, y_direction;
	// x_direction << obj.r_margin.normalized();
	x_direction << (tmp.head(3) - obj.origin).normalized();
	y_direction << -obj.o_margin.normalized().cross(x_direction);
	_Tug_normal = CController::R3D(obj, x_direction, 0);
	_Tug_normal_inv = CController::R3D(obj, x_direction, 0);

	_Tug_y = CController::R3D(obj, y_direction, 0);
	_Tug_y_inv = CController::R3D(obj, y_direction, 0);

	// cout<<"_Tug normal :\n"<<_Tug_normal<<endl;

	// _Tue <<   _Tug*_Tge;
	_Tue << _Tug_normal * _Tug_normal_inv.inverse() * _Tug_y * _Tug_y_inv.inverse() * _Tug * _Tge;
	// _Tue << _Tug*_Tge * _Tug_y_inv.inverse()*_Tug_y *_Tug_normal_inv.inverse()* _Tug_normal;
	// cout<<"rotation to euler 1:"<<CustomMath::GetBodyRotationAngle(Tvr.block(0,0,3,3)*_Tue.inverse()).transpose()<<endl;
	// cout<<"rotation to euler 2:"<<CustomMath::GetBodyRotationAngle(_Tue.inverse()).transpose()<<endl;
	// cout<<"rotation to euler 3:"<<CustomMath::GetBodyRotationAngle(_Tue).transpose()<<endl;

	target.yaw = atan2(_Tue(1, 0), _Tue(0, 0)) + robot.zrot;
	target.pitch = atan2(-_Tue(2, 0), sqrt(pow(_Tue(2, 1), 2) + pow(_Tue(2, 2), 2)));
	target.roll = atan2(_Tue(2, 1), _Tue(2, 2));

	target.yaw = fmod(target.yaw + M_PI, 2 * M_PI);
	if (target.yaw < 0)
	{
		target.yaw += 2 * M_PI;
	}
	target.yaw = target.yaw - M_PI;

	target.pitch = fmod(target.pitch + M_PI, 2 * M_PI);
	if (target.pitch < 0)
	{
		target.pitch += 2 * M_PI;
	}
	target.pitch = target.pitch - M_PI;

	target.roll = fmod(target.roll + M_PI, 2 * M_PI);

	if (target.roll < 0)
	{
		target.roll += 2 * M_PI;
	}
	target.roll = target.roll - M_PI;

	target.gripper = _gripper_close;
	target.time = 0.5;

	target.target_velocity << 0.02, 0.02, 0.02;
	target.state = "taskspace";

	return target;
}

void CController::TargetPlan()
{
	double motion_time_const = 10.0;
	double motion_time;
	double init_theta = 0.0;

	Target open_latch;
	open_latch.state = "open_latch";

	Target open_door;
	open_door.state = "open_door";
	// Target stay;
	// open_door.state = "stay";

	Target open_cabinet;
	open_cabinet.state = "open_cabinet";

	Target home;

	home.time = -1;

	// Objects obj_above = _latch;
	// obj_above.o_margin = obj_above.o_margin + obj_above.o_margin.normalized() * 0.05;
	Objects obj_above = _cabinet;

	Target _reach_cabinet1;
	_reach_cabinet1.x = obj_above.pos[0];
	_reach_cabinet1.y = obj_above.pos[1];
	_reach_cabinet1.z = obj_above.pos[2];
	_reach_cabinet1.roll = -M_PI/2;
	_reach_cabinet1.pitch = M_PI/4;
	_reach_cabinet1.yaw = -M_PI/2;
	_reach_cabinet1.gripper = _gripper_open;
	_reach_cabinet1.time = 3.0;
	_reach_cabinet1.target_velocity << 0.02, 0.02, 0.02;
	_reach_cabinet1.state = "taskspace";
	_target_plan.push_back(_reach_cabinet1);

	_reach_cabinet1.gripper = _gripper_close;
	_reach_cabinet1.time = 1.0;
	_target_plan.push_back(_reach_cabinet1);
	
	// RL
	_reach_cabinet1.state = "open";
	_reach_cabinet1.time = abs(motion_time_const * abs(_goal_theta_cabinet)) * 2;
	_target_plan.push_back(_reach_cabinet1);

	// Manual
	// _reach_cabinet1.x = obj_above.pos[0] - 0.5;
	// _reach_cabinet1.time = 5.0;
	// _target_plan.push_back(_reach_cabinet1);






	// cout << "open_cabinet.roll: " << obj_above.roll << endl;
	// cout << "open_cabinet.pitch: " << obj_above.pitch << endl;
	// cout << "open_cabinet.yaw: " << obj_above.yaw << endl;

	// _target_plan.push_back(TargetTransformMatrix(_latch, _robot, init_theta));
	// _target_plan.back().gripper = 0.04;
	// _target_plan.back().time = 1.0;
	// _target_plan.back().state = "grasp_valve";

	// _target_plan.push_back(TargetTransformMatrix(_latch, _robot, init_theta));
	// _target_plan.back().gripper = _gripper_close;
	// _target_plan.back().time = 1.0;
	// _target_plan.back().state = "grasp_valve";

	// _target_plan.push_back(open_latch);
	// motion_time = abs(motion_time_const * abs(_goal_theta_latch - init_theta) * _latch.r_margin.norm());
	// _target_plan.back().time = motion_time;
	// _target_plan.back().gripper = _gripper_close;

	// // _target_plan.push_back(stay);
	// // _target_plan.back().time = 5;
	// // _target_plan.back().gripper = _gripper_close;

	// _target_plan.push_back(open_door);
	// motion_time = abs(motion_time_const * abs(_goal_theta_door - init_theta) * _door.r_margin.norm());
	// _target_plan.back().time = motion_time;
	// _target_plan.back().gripper = _gripper_close;

	// _target_plan.push_back(TargetTransformMatrix(obj_above, _robot, _goal_theta_door)); // 제자리에서 gripper만 열도록 만들어야함.
	// _target_plan.back().gripper = 0.04;
	// _target_plan.back().time = .5;

	_target_plan.push_back(home);

}

void CController::motionPlan()
{

	if (_init_mp) // after initialize: TRUE
	{

		Robot robot_base;
		Objects cabinet;

		robot_base.pos << _robot_base;
		robot_base.zrot = 0; // M_PI;
		robot_base.ee_align = DEG2RAD * (45);

		_gripper_close = 0.00;

		cabinet.name="CABINET";
		cabinet.o_margin << -1e4, 0, 0;
		// cabinet.o_margin = _rotation_cabinet * cabinet.o_margin;
		cabinet.r_margin << 0, -1e4, 0;
		// cabinet.r_margin = _rotation_cabinet * cabinet.r_margin;
		cabinet.grab_dir << cabinet.o_margin.cross(cabinet.r_margin);
		cabinet.pos << _position_cabinet;
		cabinet.norm_dir << cabinet.o_margin;
		cabinet.origin = cabinet.pos;

		_robot = robot_base;
		
		_cabinet = cabinet;

		_grab_vector = cabinet.grab_dir.normalized();
		_normal_vector = -cabinet.o_margin.normalized();

		
		Vector4d xaxis;
		Vector4d yaxis;
		Vector4d zaxis;
		Vector4d origin;
		Vector4d porg;
		// Matrix4d _Tvb; // valve handle -> valve base
		// Matrix4d Tbu; // valve base -> universal
		// Matrix4d Tur; // universal -> robot
		// Matrix4d Tvr; // valve handle -> valve vase -> universal -> robot!!

		xaxis << cabinet.r_margin.normalized(), 0;
		yaxis << cabinet.o_margin.normalized().cross(cabinet.r_margin.normalized()), 0;
		zaxis << cabinet.o_margin.normalized(), 0;
		porg << cabinet.o_margin, 1;

		_Tvb << xaxis, yaxis, zaxis, porg;

		_Tbu << 1, 0, 0, cabinet.pos(0),
			0, 1, 0, cabinet.pos(1),
			0, 0, 1, cabinet.pos(2),
			0, 0, 0, 1;

		_Tur << cos(-robot_base.zrot), sin(-robot_base.zrot), 0, robot_base.pos(0),
			-sin(-robot_base.zrot), cos(-robot_base.zrot), 0, robot_base.pos(1),
			0, 0, 1, -robot_base.pos(2),
			0, 0, 0, 1;

		_Tvr << _Tur * _Tbu * _Tvb;

		_Rvu = _Tvr.block<3, 3>(0, 0);
		_Ruv = _Rvu.inverse();
		_Ruv_J.block<3, 3>(0, 0) = _Ruv;
		_Ruv_J.block<3, 3>(3, 3) = _Ruv;
		_Rvu_J = _Ruv_J.transpose();

		TargetPlan();

		_init_mp = false;
	}

	// Target target;
	if (_bool_plan(_cnt_plan) == 1)
	{

		if (_cnt_plan > _target_plan.size())
		{

			printf("plan terminated\n\n");
			reset_target(100000000, _q);
		}

		else
		{

			_q_goal_data.push_back(_q);
			// target.clear();
			target = _target_plan[_cnt_plan];
			_gripper_goal = target.gripper;
			if (target.state == "jointspace")
			{

				reset_target(target.time, target.q_goal);
			}
			else if (target.state == "open"){
				_pos_goal_hand(0) = target.x;
				_pos_goal_hand(1) = target.y;
				_pos_goal_hand(2) = target.z;
				_rpy_goal_hand(0) = target.roll;
				_rpy_goal_hand(1) = target.pitch;
				_rpy_goal_hand(2) = target.yaw;

				// _gripper_goal= target.gripper;
				_time_plan(_cnt_plan) = target.time;

				// reset_target(target.time, _pos_goal_hand, _rpy_goal_hand, target.state); //control mode = 5 -> RL action
				reset_target(target.time, _pos_goal_hand, _rpy_goal_hand);
				_control_mode = 3;
			}
			else if ((target.state == "taskspace") || (target.state == "grasp_valve"))
			{
				_pos_goal_hand(0) = target.x;
				_pos_goal_hand(1) = target.y;
				_pos_goal_hand(2) = target.z;
				_rpy_goal_hand(0) = target.roll;
				_rpy_goal_hand(1) = target.pitch;
				_rpy_goal_hand(2) = target.yaw;

				// _gripper_goal= target.gripper;
				_time_plan(_cnt_plan) = target.time;

				// reset_target(target.time, _pos_goal_hand, _rpy_goal_hand, target.state); //control mode = 5 -> RL action
				reset_target(target.time, _pos_goal_hand, _rpy_goal_hand); // control mode = 2 -> planned trajectory
			}
			else if (target.state == "stay")
			{
				_pos_goal_hand << _x_hand.head(3);
				_rpy_goal_hand << _x_hand.tail(3);
				_time_plan(_cnt_plan) = target.time;
				reset_target(target.time, _pos_goal_hand, _rpy_goal_hand); // control mode = 2 -> planned trajectory
			}
		}
		_cnt_plan = _cnt_plan + 1;
	}
}

void CController::reset_target(double motion_time, VectorXd target_joint_position)
{
	_control_mode = 1;
	_motion_time = motion_time;
	_bool_joint_motion = false;
	_bool_ee_motion = false;

	_q_goal = target_joint_position.head(7);
	_qdot_goal.setZero();
}

void CController::reset_target(double motion_time, Vector3d target_pos, Vector3d target_ori)
{
	_control_mode = 2;
	_motion_time = motion_time;
	_bool_joint_motion = false;
	_bool_ee_motion = false;

	_x_goal_hand.head(3) = target_pos;
	_x_goal_hand.tail(3) = target_ori;
	_xdot_goal_hand.setZero();
}

void CController::reset_target(double motion_time, string state)
{
	if (state == "open_latch")
	{
		_control_mode = 3;
	}
	else if (state == "open_door")
	{
		_control_mode = 4;
	}

	_bool_joint_motion = false;
	_bool_ee_motion = false;
	_motion_time = motion_time;
}

void CController::JointControl()
{
	_torque.setZero();
	_torque = Model._A * (400 * (_q_des - _q) + 40 * (_qdot_des - _qdot)) + Model._bg;
	// cout << _torque << endl;
}

void CController::GripperControl()
{
	// position control with mujoco
	du = _t - _start_time;
	if (_motion_time == 0.0)
	{
		_motion_time = 0.5;
	}
	if (du >= (_motion_time))
	{
		du = _motion_time;
	}
	if (_control_mode >= 3)
	{ // onvalve stay grasp
		_gripper_des = _gripper_goal;
	}
	else
	{
		_gripper_des = _init_gripper + (_gripper_goal - _init_gripper) * du / _motion_time;
	}
	_grippertorque = _kpj_gripper * (_gripper_des - _gripper) - _kdj_gripper * (_gripperdot); // PD simple damping control (_gripperdot_goal = 0 0)
}

// Closed Loop Inverse Kinematics
void CController::CLIK()
{
	_torque.setZero();

	_x_err_hand.segment(0, 3) = _x_des_hand.head(3) - _x_hand.head(3);

	_x_err_hand.segment(3, 3) = -CustomMath::getPhi(Model._R_hand, _R_des_hand);

	_J_bar_hands = CustomMath::pseudoInverseQR(_J_hands);
	_qdot_des = _J_bar_hands * (_xdot_des_hand + _x_kp * (_x_err_hand)); // + _x_err_hand.norm()*_x_force);
	_q_des = _q_des + _dt * _qdot_des;

	_torque = Model._A * (_kpj * (_q_des - _q) + _kdj * (_qdot_des - _qdot)) + Model._bg;

	_accum_err_q = _accum_err_q + (_q - _q_des).cwiseAbs();
	_accum_err_x = _accum_err_x + (_x_hand - _x_des_hand).cwiseAbs();

	_Rdot_des_hand = CustomMath::GetBodyRotationMatrix(_xdot_des_hand(3), _xdot_des_hand(4), _xdot_des_hand(5));
	_Rdot_hand = CustomMath::GetBodyRotationMatrix(_xdot_hand(3), _xdot_hand(4), _xdot_hand(5));

	_xdot_err_hand.segment(0, 3) = _xdot_des_hand.head(3) - _xdot_hand.head(3);
	_xdot_err_hand.segment(3, 3) = -CustomMath::getPhi(_Rdot_hand, _Rdot_des_hand);
}

void CController::OperationalSpaceControl()
{
	// cout<<"_xdot :"<<_xdot_des_hand.transpose()<<endl;
	_J_bar_hands = CustomMath::pseudoInverseQR(_J_hands);
	_lambda = CustomMath::pseudoInverseQR(_J_hands.transpose()) * Model._A * _J_bar_hands;

	_x_err_hand.segment(0, 3) = _x_des_hand.head(3) - _x_hand.head(3);
	_x_err_hand.segment(3, 3) = -CustomMath::getPhi(_R_hand, _R_des_hand);

	_Rdot_des_hand = CustomMath::GetBodyRotationMatrix(_xdot_des_hand(3), _xdot_des_hand(4), _xdot_des_hand(5));
	_Rdot_hand = CustomMath::GetBodyRotationMatrix(_xdot_hand(3), _xdot_hand(4), _xdot_hand(5));

	_xdot_err_hand.segment(0, 3) = _xdot_des_hand.head(3) - _xdot_hand.head(3);
	_xdot_err_hand.tail(3) = -_xdot_hand.tail(3);
	_force = _kpj * _x_err_hand + _kdj * _xdot_err_hand;

	_torque = _J_hands.transpose() * _lambda * _force + Model._bg;
}

void CController::HybridControl()
{
	if ((target.state == "taskspace") || (target.state == "stay"))
	{
		CLIK();
	}

	else if ((target.state == "open"))
	{
		_J_bar_hands = CustomMath::pseudoInverseQR(_J_hands);
		_lambda = CustomMath::pseudoInverseQR((_Ruv_J * _J_hands).transpose()) * Model._A * CustomMath::pseudoInverseQR(_Ruv_J * _J_hands);
		MatrixXd lambda_diagonal = _lambda.diagonal().asDiagonal();
		VectorXd x_des_handV, x_handV, xdot_des_handV, xdot_handV;
		Matrix3d R_des_handV, R_handV, Rdot_des_handV, Rdot_handV, skew;
		x_des_handV.setZero(6);
		x_handV.setZero(6);
		xdot_des_handV.setZero(6);
		xdot_handV.setZero(6);

		x_des_handV.head(3) = _Ruv * _x_des_hand.head(3);
		x_handV.head(3) = _Ruv * _x_hand.head(3);
		R_des_handV = _Ruv * _R_des_hand;
		// R_des_handV = _Ruv * _R_fix;

		R_handV = _Ruv * _R_hand;
		// Rdot_des_handV = CustomMath::GetBodyRotationMatrix(xdot_des_handV(3), xdot_des_handV(4), xdot_des_handV(5));

		xdot_des_handV = _Ruv_J * _xdot_des_hand;
		xdot_handV = _Ruv_J * _xdot_hand;
		Rdot_handV = CustomMath::GetBodyRotationMatrix(xdot_handV(3), xdot_handV(4), xdot_handV(5));

		_x_err_hand.head(3) = x_des_handV.head(3) - x_handV.head(3);
		_xdot_err_hand.head(3) = xdot_des_handV.head(3) - xdot_handV.head(3);

		_x_err_hand.tail(3) = -CustomMath::getPhi(R_handV, R_des_handV);
		_xdot_err_hand.tail(3) << 0, 0, 0;

		_force = _kpj * _x_err_hand + _kdj * _xdot_err_hand;
		
		MatrixXd SelectionV;
		SelectionV.setZero(6, 6);
		// motion control selection matrix
		MatrixXd SelectionX;
		SelectionX.setIdentity(6, 6);
		VectorXd FORCEX;
		FORCEX.setZero(6);
		VectorXd FORCEV;
		FORCEV.setZero(6);

		VectorXd FORCE;
		FORCE.setZero(6);

		SelectionV.diagonal() << 0, 0, 1, 0, 0, 0;
		SelectionX = SelectionX - SelectionV;

		_Force = SelectionV.diagonal() * _force_gain;
		

		// _Force(2) = 0;
		FORCEV = _Rvu_J * (SelectionV * _Force);
		VectorXd selected_force;
		selected_force.setZero(6);
		VectorXd transformed_force;
		transformed_force.setZero(6);
		MatrixXd selected_rot;
		selected_rot.setZero(3, 3);

		MatrixXd gain;
		gain.setZero(6, 6);
		// gain.diagonal() << 0,0,5,5,5,5;
		selected_force << SelectionX * _force;

		// cout<<_force(0)<<","<<_force(1)<<","<<sqrt(_force(0)*_force(0) + _force(1)*_force(1))<<endl;

		// FORCEX = _lambda * _Rvu_J * selected_force;
		FORCEX = _Rvu_J * _lambda * selected_force;
		// FORCEX = _Rvu_J * lambda_diagonal * selected_force;
		// if (target.state != "grasp_valve" ){
		// 	FORCEX << 0.,0.,0.,1.,1.,0.;
		// }
		// cout<<(FORCEX).transpose()<<endl;
		FORCE << FORCEX + FORCEV;

		_torque = _J_hands.transpose() * FORCE + Model._bg;

		// cout << RED << "SelectionX: \n" << SelectionX << RESET << endl;
		// cout << YELLOW << "_force: \n" << _force << RESET << endl;
		// cout << "====================================" << endl;

		// cout << "selected force :" << selected_force.transpose() << endl;
		// cout << _Rvu_J << endl;
		// cout << _lambda << endl;
		// cout << "  _FORCE :" << FORCE.transpose() << endl;

		// cout << "  _FORCEV :" << FORCEV.transpose() << endl;

		// cout << "  _FORCEX :" << FORCEX.transpose() << endl;
		// cout<<"rotation error base  "<<-CustomMath::getPhi(_R_hand, _R_des_hand).transpose()<<endl;

		// cout<<"rotation error valve "<<-CustomMath::getPhi(R_handV, R_des_handV).transpose()<<endl;
		// cout<<"xdot_hand "<<_xdot_hand.transpose()<<endl;
		// cout<<"xdot_handV "<<xdot_handV.transpose()<<endl;

		// cout<<"force :"<<_force.transpose()<<endl;

		// cout<<"FORCEX:"<<FORCEX.transpose()<<endl;
		// cout<<"FORCEV:"<<FORCEV.transpose()<<endl;

		// _torque = _J_hands.transpose() * _lambda*FORCE + Model._bg;

		// _torque = _J_hands.transpose() * _lambda.diagonal().cwiseProduct(_force) + Model._bg;
	}
	else
	{
		_torque = _J_hands.transpose() * _lambda * _force + Model._bg;

		// x_hand_fix = _x_hand;
		_R_fix = _R_des_hand;
	}
}

void CController::Initialize()
{

	_control_mode = 1; // 1: joint space, 2: task space(CLIK)
	_gripper_mode = 1;
	_init_mp = true;
	_bool_init = true;
	_t = 0.0;
	_init_t = 0.0;
	_pre_t = 0.0;
	_dt = 0.0;

	// _kpj = 100;
	// _kdj = 10;
	_kpj = 400;
	_kdj = 40;
	_kpj_gripper = 30000.0; // 100;
	_kdj_gripper = 10.0;	// 10;

	_x_kp = 0.1; // 작게 0.1

	_q.setZero(_k);
	_qdot.setZero(_k);
	_torque.setZero(_k);
	_gripper = 0;
	_gripperdot = 0; // gripper 속도 : 50mm/s
	_grippertorque = 0;

	_robot_base.setZero(3);
	_R_fix.setZero(3, 3);

	_J_hands.setZero(6, _k);
	_J_bar_hands.setZero(_k, 6);

	_x_hand.setZero(6);
	_xdot_hand.setZero(6);

	_cnt_plan = 0;
	_bool_plan.setZero(100);
	_time_plan.resize(100);
	_time_plan.setConstant(-1);

	_q_home.setZero(_k);

	_q_home(0) = 0.0;
	_q_home(1) = -60 * DEG2RAD; // -0.7853981633974483; //-45
	_q_home(2) = 0.0;
	_q_home(3) = -90 * DEG2RAD; //-2.356194490192345; //-135
	_q_home(4) = 0.0;
	_q_home(5) = 90 * DEG2RAD; // 1.5707963267948966; // 90
	_q_home(6) = 45 * DEG2RAD; // 0.7853981633974483; // 45

	_gripper_close = 0.0;
	_gripper_open = 0.03;

	_kpj_diagonal.setZero(_k, _k);
	//							0 		1	2		3	   4	5 	6
	_kpj_diagonal.diagonal() << 400., 2500., 3000., 1700., 700., 500., 520.; // armarture=0.1 이었을 때의 gain / inertial사용 (x)
	// _kpj_diagonal.diagonal() << 20., 40., 40., 20., 20., 20., 20.;

	_kdj_diagonal.setZero(_k, _k);
	_kdj_diagonal.diagonal() << 20., 250., 170., 320., 70., 50., 15.;
	// _kdj_diagonal.diagonal() << 2., 4., 4., 2., 2., 2., 2.;

	_kpj_diagonal6.setZero(6, 6);
	_kpj_diagonal6.diagonal() << 400., 400., 400., 50., 50., 50.; // armarture=0.1 이었을 때의 gain / inertial사용 (x)

	_kdj_diagonal6.setZero(6, 6);
	_kdj_diagonal6.diagonal() << 40., 40., 40., 5., 5., 5.;

	_grab_vector.setZero(3);
	_normal_vector.setZero(3);
	_origin.setZero(3);
	_radius = 0.0;
	_goal_theta = 0.0;
	_init_theta = 0.0;
	_drpy.setZero(3);
	_dxyz.setZero(3);

	_start_time = 0.0;
	_end_time = 0.0;
	_motion_time = 0.0;

	_bool_joint_motion = false;
	_bool_ee_motion = false;
	// _q_pre.
	_q_des.setZero(_k);
	_qdot_des.setZero(_k);
	_q_pre.setZero(_k);
	_qdot_pre.setZero(_k);
	_q_goal.setZero(_k);
	_qdot_goal.setZero(_k);
	_gripper_des = 0;
	_gripper_goal = 0;
	_gripperdot_goal = 0;
	_init_gripper = 0;

	// _gripper_des.setZero(2);
	// _gripper_goal.setZero(2);
	// _gripperdot_goal.setZero(2);
	// _init_gripper.setZero(2);

	_x_des_hand.setZero(6);
	_xdot_des_hand.setZero(6);
	_x_goal_hand.setZero(6);
	_xdot_goal_hand.setZero(6);

	_pos_goal_hand.setZero(); // 3x1
	_rpy_goal_hand.setZero(); // 3x1

	JointTrajectory.set_size(_k);
	_A_diagonal.setZero(_k, _k);
	torque_command.clear();

	_x_err_hand.setZero(6);
	_xdot_err_hand.setZero(6);
	_R_des_hand.setZero();
	_R_hand.setZero(3, 3);
	_Rdot_des_hand.setZero();
	_Rdot_hand.setZero();
	_lambda.setZero(6, 6);
	_force.setZero(6);
	_compensate_force.setZero(6);
	_compensated_force.setZero(6);
	_Force.setZero(6);
	_I.setIdentity(7, 7);
	_rotation_obj.setIdentity(3, 3);
	_Tvr.setIdentity(4, 4);
	_Tvb.setIdentity(4, 4);
	_Tbu.setIdentity(4, 4);
	_Tur.setIdentity(4, 4);

	du = 0.0;

	_print_time = 0;
	_print_interval = 0.1;
	_target_plan.clear();
	_q_goal_data.clear();
	dr.clear();
	dp.clear();
	dy.clear();

	_x_plan.clear();
	_y_plan.clear();
	_z_plan.clear();
	_theta_des = 0;
	accum_err.setZero(7);
	_accum_err_x.setZero(6);
	_accum_err_q.setZero(7);
	_x_force.setZero(6);
	_rpy_des.setZero(3);

	_scale_obj = 7;
	_generate_dxyz = true;
	_Rvu_J.setZero(6, 6);
	_Ruv_J.setZero(6, 6);
	_Rvu.setZero(3, 3);
	_Ruv.setZero(3, 3);
	_x_circular_hand.setZero(6);

	_cnt_plot = 0;
	_q_plot.setZero(10000, 7);
	_qdot_plot.setZero(10000, 7);
	_x_plot.setZero(10000, 6);

	_eye.setIdentity(7, 7);
	_Null.setZero(7, 7);
	_qnull_err.setZero(7);
	_q_mid.setZero(7);
	_q_mid << 0, 0, 0, (-0.1518 - 3.0421) * 0.5, 0, (4.5169, 0.5445) * 0.5, 0;

	_force_gain = 0;
	_dforce_gain = 0;
	_dtheta_des = 0.001;

	_roll = 0.;
	_pitch = 0.;
	_droll = 0.;
	_dpitch = 0.;
	_ddroll = 0.;
	_ddpitch = 0.;
	_kf = -1.;

	_q_valve = 0.;
	_qdot_valve = 0.;

	_R_hand_ref.setZero(3,3);

	_robot_base << 0, 0, 0;
	_x_tmp.setZero(6);

	action_torque.setZero(7);

	load_model();
}

void CController::Initialize_Cabinet(int planning_mode, array<array<double, 4>, 4> cabinet_info, double goal_theta)
{

	_control_mode = 1; // 1: joint space, 2: task space(CLIK)
	_gripper_mode = 1;
	_init_mp = true;
	_bool_init = true;
	_t = 0.0;
	_init_t = 0.0;
	_pre_t = 0.0;
	_dt = 0.0;

	// _kpj = 100;
	// _kdj = 10;
	_kpj = 400;
	_kdj = 40;
	_kpj_gripper = 30000.0; // 100;
	_kdj_gripper = 10.0;	// 10;

	_x_kp = 0.1; // 작게 0.1

	_q.setZero(_k);
	_qdot.setZero(_k);
	_torque.setZero(_k);
	_gripper = 0;
	_gripperdot = 0; // gripper 속도 : 50mm/s
	_grippertorque = 0;

	_planning_mode = planning_mode;
	_robot_base.setZero(3);
	_R_fix.setZero(3, 3);

	_J_hands.setZero(6, _k);
	_J_bar_hands.setZero(_k, 6);

	_x_hand.setZero(6);
	_xdot_hand.setZero(6);

	_cnt_plan = 0;
	_bool_plan.setZero(100);
	_time_plan.resize(100);
	_time_plan.setConstant(-1);

	_q_home.setZero(_k);

	_q_home(0) = 0.0;
	_q_home(1) = -60 * DEG2RAD; // -0.7853981633974483; //-45
	_q_home(2) = 0.0;
	_q_home(3) = -90 * DEG2RAD; //-2.356194490192345; //-135
	_q_home(4) = 0.0;
	_q_home(5) = 90 * DEG2RAD; // 1.5707963267948966; // 90
	_q_home(6) = 45 * DEG2RAD; // 0.7853981633974483; // 45

	_gripper_close = 0.0;
	_gripper_open = 0.03;

	_kpj_diagonal.setZero(_k, _k);
	//							0 		1	2		3	   4	5 	6
	_kpj_diagonal.diagonal() << 400., 2500., 3000., 1700., 700., 500., 520.; // armarture=0.1 이었을 때의 gain / inertial사용 (x)
	// _kpj_diagonal.diagonal() << 20., 40., 40., 20., 20., 20., 20.;

	_kdj_diagonal.setZero(_k, _k);
	_kdj_diagonal.diagonal() << 20., 250., 170., 320., 70., 50., 15.;
	// _kdj_diagonal.diagonal() << 2., 4., 4., 2., 2., 2., 2.;

	_kpj_diagonal6.setZero(6, 6);
	_kpj_diagonal6.diagonal() << 400., 400., 400., 50., 50., 50.; // armarture=0.1 이었을 때의 gain / inertial사용 (x)

	_kdj_diagonal6.setZero(6, 6);
	_kdj_diagonal6.diagonal() << 40., 40., 40., 5., 5., 5.;

	_grab_vector.setZero(3);
	_normal_vector.setZero(3);
	_origin.setZero(3);
	_radius = 0.0;
	_goal_theta = 0.0;
	_init_theta = 0.0;
	_drpy.setZero(3);
	_dxyz.setZero(3);

	_start_time = 0.0;
	_end_time = 0.0;
	_motion_time = 0.0;

	_bool_joint_motion = false;
	_bool_ee_motion = false;
	// _q_pre.
	_q_des.setZero(_k);
	_qdot_des.setZero(_k);
	_q_pre.setZero(_k);
	_qdot_pre.setZero(_k);
	_q_goal.setZero(_k);
	_qdot_goal.setZero(_k);
	_gripper_des = 0;
	_gripper_goal = 0;
	_gripperdot_goal = 0;
	_init_gripper = 0;

	// _gripper_des.setZero(2);
	// _gripper_goal.setZero(2);
	// _gripperdot_goal.setZero(2);
	// _init_gripper.setZero(2);

	_x_des_hand.setZero(6);
	_xdot_des_hand.setZero(6);
	_x_goal_hand.setZero(6);
	_xdot_goal_hand.setZero(6);

	_pos_goal_hand.setZero(); // 3x1
	_rpy_goal_hand.setZero(); // 3x1

	JointTrajectory.set_size(_k);
	_A_diagonal.setZero(_k, _k);
	torque_command.clear();

	_x_err_hand.setZero(6);
	_xdot_err_hand.setZero(6);
	_R_des_hand.setZero();
	_R_hand.setZero(3, 3);
	_Rdot_des_hand.setZero();
	_Rdot_hand.setZero();
	_lambda.setZero(6, 6);
	_force.setZero(6);
	_compensate_force.setZero(6);
	_compensated_force.setZero(6);
	_Force.setOnes(6);
	_I.setIdentity(7, 7);
	_rotation_obj.setIdentity(3, 3);
	_Tvr.setIdentity(4, 4);
	_Tvb.setIdentity(4, 4);
	_Tbu.setIdentity(4, 4);
	_Tur.setIdentity(4, 4);

	du = 0.0;

	_print_time = 0;
	_print_interval = 0.1;
	_target_plan.clear();
	_q_goal_data.clear();
	dr.clear();
	dp.clear();
	dy.clear();

	_x_plan.clear();
	_y_plan.clear();
	_z_plan.clear();
	_theta_des = 0;
	accum_err.setZero(7);
	_accum_err_x.setZero(6);
	_accum_err_q.setZero(7);
	_x_force.setZero(6);
	_rpy_des.setZero(3);

	_scale_obj = 7;
	_generate_dxyz = true;
	_Rvu_J.setZero(6, 6);
	_Ruv_J.setZero(6, 6);
	_Rvu.setZero(3, 3);
	_Ruv.setZero(3, 3);
	_x_circular_hand.setZero(6);

	_cnt_plot = 0;
	_q_plot.setZero(10000, 7);
	_qdot_plot.setZero(10000, 7);
	_x_plot.setZero(10000, 6);

	_eye.setIdentity(7, 7);
	_Null.setZero(7, 7);
	_qnull_err.setZero(7);
	_q_mid.setZero(7);
	_q_mid << 0, 0, 0, (-0.1518 - 3.0421) * 0.5, 0, (4.5169, 0.5445) * 0.5, 0;

	_force_gain = 0;
	_dforce_gain = 0;
	_dtheta_des = 0.001;

	_roll = 0.;
	_pitch = 0.;
	_droll = 0.;
	_dpitch = 0.;
	_ddroll = 0.;
	_ddpitch = 0.;
	_kf = -1.;

	_q_valve = 0.;
	_qdot_valve = 0.;
	_rotation_latch.setZero(3, 3);
	_rotation_door.setZero(3, 3);
	_position_latch.setZero(3);
	_position_door.setZero(3);
	// _goal_theta_latch = goal_theta[0];
	// _goal_theta_door = goal_theta[1];

	// for cabinet
	_position_cabinet.setZero(3);
	_rotation_cabinet.setZero(3, 3);
	_goal_theta_cabinet = goal_theta;

	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			_rotation_cabinet(i, j) = cabinet_info[i][j];
		}
	}
	for (int i = 0; i < 3; i++)
	{
		_position_cabinet(i) = cabinet_info[i][3];
	}
	_position_cabinet[0] += 0.001;
	_robot_base << 0, 0, 0;
	_x_tmp.setZero(6);

	load_model();
}

namespace py = pybind11;
PYBIND11_MODULE(controller, m)
{
	m.doc() = "pybind11 for controller";

	py::class_<CController>(m, "CController")
		.def(py::init<int>())
		.def("read", &CController::read_pybind)
		.def("control_mujoco", &CController::control_mujoco)
		.def("write", &CController::write_pybind)
		// .def("write_force", &CController::write_force_pybind)
		.def("initialize", &CController::Initialize)
		.def("initialize_cabinet", &CController::Initialize_Cabinet)
		.def("put_action", &CController::put_action_pybind)
		.def("relative_T_hand", &CController::relative_T_hand_pybind)
		// .def("randomize_env", &CController::randomize_env_pybind)
		.def("get_ee", &CController::get_ee_pybind)
		.def("get_jacobian", &CController::get_jacobian_pybind)
		.def("get_model", &CController::get_model_pybind)
		.def("control_mode", &CController::control_mode_pybind)
		.def("write_control_mode", &CController::write_control_mode_pybind)
		.def("desired_rpy", &CController::desired_rpy_pybind)
		.def("get_commands", &CController::get_commands_pybind)
		// .def("target_replan", &CController::TargetRePlan_pybind)
		// .def("target_replan2", &CController::TargetRePlan2_pybind)
		// .def("target_replan3", &CController::TargetRePlan3_pybind)
		.def("get_action", &CController::get_actions_pybind)
		.def("get_force_gain", &CController::get_force_gain)
		;
		// .def("get_commands", &CController::get_commands_pybind);

	//   .def("write", &CController::write);

#ifdef VERSION_INFO
	m.attr("__version__") = VERSION_INFO;
#else
	m.attr("__version__") = "dev";
#endif
	//   m.attr("TEST") = py::int_(int(42));
}