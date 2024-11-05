#include "controller.h"

CController::CController()
{
	_k = 7;
	Initialize();
}

CController::~CController()
{
	
}

void CController::Initialize()
{
    _control_mode = 1; //1: joint space, 2: task space(CLIK), 3: operational space

	_bool_init = true;
	_t = 0.0;
	_init_t = 0.0;
	_pre_t = 0.0;
	_dt = 0.0;

	_kpj = 400.0;
	_kdj = 20.0;

	_x_kp = 1;
	_x_kd = 1;

    _q.setZero(_k);
	_qdot.setZero(_k);
	_torque.setZero(_k);

	_J_hands.setZero(6,_k);
	_J_bar_hands.setZero(_k,6);
	_J_T_hands.setZero(_k, 6);

	_x_hand.setZero(6);
	_xdot_hand.setZero(6);

	_bool_plan.setZero(30);

	_q_home.setZero(_k);
	_q_home(0) = 0;
	_q_home(1) = -M_PI_4;
	_q_home(2) = 0;
	_q_home(3) = -3 * M_PI_4;
	_q_home(4) = 0;
	_q_home(5) = M_PI_2;
	_q_home(6) = M_PI_4;

	_start_time = 0.0;
	_end_time = 0.0;
	_motion_time = 0.0;

	_bool_joint_motion = false;
	_bool_ee_motion = false;

	_q_des.setZero(_k);
	_qdot_des.setZero(_k);
	_q_goal.setZero(_k);
	_qdot_goal.setZero(_k);

	_x_des_hand.setZero(6);
	_xdot_des_hand.setZero(6);
	_x_goal_hand.setZero(6);
	_xdot_goal_hand.setZero(6);

	_pos_goal_hand.setZero(); // 3x1 
	_rpy_goal_hand.setZero(); // 3x1
	JointTrajectory.set_size(_k);
	_A_diagonal.setZero(_k,_k);

	_x_err_hand.setZero(6);
	_x_dot_err_hand.setZero(6);
	_R_des_hand.setZero();

	_I.setIdentity(7,7);
	_J_null.setZero(_k,_k);

	_pre_q.setZero(7);
	_pre_qdot.setZero(7);

	_q_order.setZero(7);
	_qdot_order.setZero(7);

	_cnt_plan = 0;
	_bool_plan(_cnt_plan) = 1;
}

void CController::read(double t, double* q, double* qdot)
{	
	_t = t;
	if (_bool_init == true)
	{
		_init_t = _t;
		_bool_init = false;
	}

	_dt = t - _pre_t;
	_pre_t = t;

	for (int i = 0; i < _k; i++)
	{
		_q(i) = q[i];
		_qdot(i) = qdot[i];
		_pre_q(i) = _q(i);
		_pre_qdot(i) = _qdot(i);
	}
}

// for pybind11
void CController::read_pybind(double t, array<double, 9> q, array<double, 9> qdot, double timestep)
{
	_dt = timestep;
	_t = t;

	if (_bool_init == true)
	{
		_init_t = _t;
		_bool_init = false;
	}

	for (int i = 0; i < _k; i++)
	{
		_q(i) = q[i];
		_qdot(i) = qdot[i];
		_pre_q(i) = _q(i);
		_pre_qdot(i) = _qdot(i);
	}
}

void CController::write(double* torque)
{
	for (int i = 0; i < _k; i++)
	{
		torque[i] = _torque(i);
	}
}

// for pybind11
std::vector<double> CController::write_pybind()
{
	pybind_torque.clear();

	for (int i = 0; i < _k; i++)
	{
		pybind_torque.push_back(_torque(i));
	}
	for (int i = 0; i < 1; i++)
	{
		pybind_torque.push_back(0); // gripper
	}

	return pybind_torque;
}

void CController::control_mujoco()
{
    ModelUpdate();
    motionPlan();
	if(_control_mode == 1) // joint space control
	{
		if (_t - _init_t < 0.1 && _bool_joint_motion == false)
		{
			_start_time = _init_t;
			_end_time = _start_time + _motion_time;
			JointTrajectory.reset_initial(_start_time, _q, _qdot);
			JointTrajectory.update_goal(_q_goal, _qdot_goal, _end_time);
			_bool_joint_motion = true;
		}
		
		JointTrajectory.update_time(_t);
		_q_des = JointTrajectory.position_cubicSpline();
		_qdot_des = JointTrajectory.velocity_cubicSpline();

		JointControl();

		if (JointTrajectory.check_trajectory_complete() == 1)
		{
			_bool_plan(_cnt_plan) = 1;
			_bool_init = true;
		}
	}
	else if(_control_mode == 2) // inverse kinematics control (CLIK)
	{		
		if (_t - _init_t < 0.1 && _bool_ee_motion == false)
		{
			_start_time = _init_t;
			_end_time = _start_time + _motion_time;
			HandTrajectory.reset_initial(_start_time, _x_hand, _xdot_hand);
			HandTrajectory.update_goal(_x_goal_hand, _xdot_goal_hand, _end_time);
			_bool_ee_motion = true;
		}

		HandTrajectory.update_time(_t);
		_x_des_hand.head(3) = HandTrajectory.position_cubicSpline();
		_R_des_hand = HandTrajectory.rotationCubic();
		_x_des_hand.segment<3>(3) = CustomMath::GetBodyRotationAngle(_R_des_hand);
		_xdot_des_hand.head(3) = HandTrajectory.velocity_cubicSpline();
		_xdot_des_hand.segment<3>(3) = HandTrajectory.rotationCubicDot();		

		CLIK();

		if (HandTrajectory.check_trajectory_complete() == 1)
		{
			_bool_plan(_cnt_plan) = 1;
			_bool_init = true;
		}
	}
	else if(_control_mode == 3) // operational space control
	{
		if (_t - _init_t < 0.1 && _bool_ee_motion == false)
		{
			_start_time = _init_t;
			_end_time = _start_time + _motion_time;
			HandTrajectory.reset_initial(_start_time, _x_hand, _xdot_hand);
			HandTrajectory.update_goal(_x_goal_hand, _xdot_goal_hand, _end_time);
			_bool_ee_motion = true;
		}

		HandTrajectory.update_time(_t);
		_x_des_hand.head(3) = HandTrajectory.position_cubicSpline();
		_R_des_hand = HandTrajectory.rotationCubic();
		_x_des_hand.segment<3>(3) = CustomMath::GetBodyRotationAngle(_R_des_hand);
		_xdot_des_hand.head(3) = HandTrajectory.velocity_cubicSpline();
		_xdot_des_hand.segment<3>(3) = HandTrajectory.rotationCubicDot();		

		OperationalSpaceControl();

		if (HandTrajectory.check_trajectory_complete() == 1)
		{
			_bool_plan(_cnt_plan) = 1;
			_bool_init = true;
		}
	}
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
	_x_hand.tail(3) = CustomMath::GetBodyRotationAngle(Model._R_hand);
	_xdot_hand = Model._xdot_hand;
}	

void CController::motionPlan()
{	
	if (_bool_plan(_cnt_plan) == 1)
	{
		if(_cnt_plan == 0)
		{	
			// cout << "plan: " << _cnt_plan << endl;
			_q_order(0) = _q_home(0);
			_q_order(1) = _q_home(1);
			_q_order(2) = _q_home(2);
			_q_order(3) = _q_home(3);
			_q_order(4) = _q_home(4);
			_q_order(5) = _q_home(5);
			_q_order(6) = _q_home(6);		                    
			reset_target(5.0, _q_order, _qdot);
			_cnt_plan++;
		}
		else
		{
			// cout << "plan: random sampled EE" << endl;

			VectorXd target_pose;
			target_pose.setZero(6);
			target_pose = get_random_sampled_EE();

			reset_target(5.0, target_pose);
			_cnt_plan++;
		}
		
	}
}

void CController::reset_target(double motion_time, VectorXd target_joint_position, VectorXd target_joint_velocity)
{
	// joint space control
	_control_mode = 1;
	_motion_time = motion_time;
	_bool_joint_motion = false;
	_bool_ee_motion = false;

	_q_goal = target_joint_position.head(7);
	// _qdot_goal = target_joint_velocity.head(7);
	_qdot_goal.setZero();
}

void CController::reset_target(double motion_time, Vector3d target_pos, Vector3d target_ori)
{
	// inverse kinematics control
	_control_mode = 2;
	_motion_time = motion_time;
	_bool_joint_motion = false;
	_bool_ee_motion = false;

	_x_goal_hand.head(3) = target_pos;
	_x_goal_hand.tail(3) = target_ori;
	_xdot_goal_hand.setZero();
}

void CController::reset_target(double motion_time, VectorXd target_pose)
{
	// operational space control
	_control_mode = 3;
	_motion_time = motion_time;
	_bool_joint_motion = false;
	_bool_ee_motion = false;

	_x_goal_hand = target_pose;
	_xdot_goal_hand.setZero();
}

void CController::JointControl()
{	
	// _control_mode = 1
	_torque.setZero();
	_A_diagonal = Model._A;
	for(int i = 0; i < 7; i++){
		_A_diagonal(i,i) += 1.0;
	}
	// Manipulator equations of motion in joint space
	_torque = _A_diagonal*(400*(_q_des - _q) + 40*(_qdot_des - _qdot)) + Model._bg;
}

void CController::CLIK()
{
	// _control_mode = 2
	_torque.setZero();	

	_x_err_hand.segment(0,3) = _x_des_hand.head(3) - _x_hand.head(3);
	_x_err_hand.segment(3,3) = -CustomMath::getPhi(Model._R_hand, _R_des_hand);

	_J_bar_hands = CustomMath::pseudoInverseQR(_J_hands);

	_dt = 0.003;
	_qdot_des = _J_bar_hands*(_xdot_des_hand + _x_kp*(_x_err_hand));
	_q_des = _q_des + _dt*_qdot_des;
	_A_diagonal = Model._A;
	for(int i = 0; i < 7; i++){
		_A_diagonal(i,i) += 1.0;
	}

	_torque = _A_diagonal * (400 * (_q_des - _q) + 40 * (_qdot_des - _qdot)) + Model._bg;
}

void CController::OperationalSpaceControl()
{
	// _control_mode = 3
	_torque.setZero();

	// calc position, velocity errors
	_x_err_hand.segment(0,3) = _x_des_hand.head(3) - _x_hand.head(3);
	_x_err_hand.segment(3,3) = -CustomMath::getPhi(Model._R_hand, _R_des_hand);
	_x_dot_err_hand.segment(0,3) = _xdot_des_hand.head(3) - _xdot_hand.head(3);
	_x_dot_err_hand.segment(3,3) = _xdot_des_hand.tail(3) - _xdot_hand.tail(3);

	// jacobian pseudo inverse matrix
	_J_bar_hands = CustomMath::pseudoInverseQR(_J_hands);

	// jacobian transpose matrix
	_J_T_hands = _J_hands.transpose();

	// jacobian inverse transpose matrix
	_J_bar_T_hands = CustomMath::pseudoInverseQR(_J_T_hands);

	// Should consider [Null space] cuz franka_panda robot = 7 DOF
	_J_null = _I - _J_T_hands * _J_bar_T_hands;

	// Inertial matrix: operational space
	_Lambda = _J_bar_T_hands * Model._A * _J_bar_hands;

	F_command_star = 400 * _x_err_hand + 40 * _x_dot_err_hand;

	_torque = (_J_T_hands * _Lambda * F_command_star + Model._bg) + _J_null * Model._A * (_qdot_des-_qdot);
}

// for pybind11
int CController::count_plan_pybind()
{
	return _cnt_plan;
}

// for pybind11
void CController::write_qpos_init_pybind(std::array<double, 9> _q_init)
{
	for (int i = 0; i < _k; ++i)
	{
		_q_home[i] = _q_init[i];
	}
}

// for pybind11
std::array<double, 9> CController::get_joint_position_pybind()
{
	std::array<double, 9> _q_pybind;

	for (int i = 0; i < _k; ++i)
	{
		_q_pybind[i] = _q[i];
	}
	for (int i = 0; i < 2; ++i)
	{
		_q_pybind[i] = 0;
	}

	return _q_pybind;
}

// for pybind11
std::array<double, 6> CController::get_EE_pybind()
{
	std::array<double, 6> EE_pybind;

	for (int i = 0; i < 6; ++i)
	{
		EE_pybind[i] = _x_hand[i];
	}

	return EE_pybind;
}

// for pybind11
void CController::write_random_sampled_EE_pybind(std::array<double, 6> sampled_EE)
{
	for(int i = 0; i < 6; ++i)
	{
		random_sampled_EE[i] = sampled_EE[i];
	}
}

Eigen::Vector<double, 6> CController::get_random_sampled_EE()
{
	return random_sampled_EE;
}

// for pybind11
namespace py = pybind11;
PYBIND11_MODULE(controller, m)
{
	m.doc() = "pybind11 for controller";

	py::class_<CController>(m, "CController")
		.def(py::init<>())
		.def("initialize", &CController::Initialize)
		.def("read", &CController::read_pybind)
		.def("control_mujoco", &CController::control_mujoco)
		.def("write", &CController::write_pybind)
		.def("count_plan", &CController::count_plan_pybind)
		.def("write_qpos_init", &CController::write_qpos_init_pybind)
		.def("get_joint_position", &CController::get_joint_position_pybind)
		.def("get_EE", &CController::get_EE_pybind)
		.def("write_random_sampled_EE", &CController::write_random_sampled_EE_pybind)
		;

#ifdef VERSION_INFO
	m.attr("__version__") = VERSION_INFO;
#else
	m.attr("__version__") = "dev";
#endif
	//   m.attr("TEST") = py::int_(int(42));
}