#include "trajectory.h"

CTrajectory::CTrajectory()
{	
	Initialize();
}

CTrajectory::~CTrajectory()
{
}

void CTrajectory::set_size(int dof)
{
	_vector_size = dof;
	_init_pos.setZero(_vector_size);
	_init_vel.setZero(_vector_size);
	_goal_pos.setZero(_vector_size);
	_goal_vel.setZero(_vector_size);
}

void CTrajectory::Initialize()
{
	_time_start = 0.0;
	_time = 0.0;
	_time_end = 0.0;
	_vector_size = 1; //default = 1
	_init_pos.setZero(_vector_size);
	_init_vel.setZero(_vector_size);
	_goal_pos.setZero(_vector_size);
	_goal_vel.setZero(_vector_size);
	_bool_trajectory_complete = false;
	_motion_threshold = 0.0005;
}

void CTrajectory::reset_initial(double time0, VectorXd init_pos, VectorXd init_vel)
{
	check_vector_size(init_pos);
	check_vector_size(init_vel);

	_time_start = time0;
	_init_pos = init_pos;
	_init_vel = init_vel;

	_bool_trajectory_complete = false;
}


void CTrajectory::update_time(double time)
{
	_time = time;
}

void CTrajectory::update_goal(VectorXd goal_pos, VectorXd goal_vel, double goal_time)
{
	check_vector_size(goal_pos);
	check_vector_size(goal_vel);
	_goal_pos = goal_pos;
	_goal_vel = goal_vel;
	_time_end = goal_time;
}

VectorXd CTrajectory::position_cubicSpline()
{
	VectorXd xd(_vector_size);

	if (_time <= _time_start)
	{
		xd = _init_pos;
	}
	else if (_time >= _time_end)
	{
		xd = _goal_pos;
	}
	else {
		xd = _init_pos + _init_vel * (_time - _time_start)
			+ (3.0 * (_goal_pos - _init_pos) / ((_time_end - _time_start) * (_time_end - _time_start)) - 2.0 * _init_vel / (_time_end - _time_start) - _goal_vel / (_time_end - _time_start)) * (_time - _time_start) * (_time - _time_start)
			+ (-2.0 * (_goal_pos - _init_pos) / ((_time_end - _time_start) * (_time_end - _time_start) * (_time_end - _time_start)) + (_init_vel + _goal_vel) / ((_time_end - _time_start) * (_time_end - _time_start))) * (_time - _time_start) * (_time - _time_start) * (_time - _time_start);
	}

	for (int i = 0; i < _vector_size; i++) //do not use cubic spline when desired motion is small
	{
		if (abs(_goal_pos(i) - _init_pos(i)) <= _motion_threshold)
		{
			xd(i) = _goal_pos(i);
		}
	}

	return xd;
}

VectorXd CTrajectory::velocity_cubicSpline()
{
	VectorXd xdotd(_vector_size);

	if (_time <= _time_start)
	{
		xdotd = _init_vel;
	}
	else if (_time >= _time_end)
	{
		xdotd = _goal_vel;
	}
	else {
		xdotd = _init_vel + 2.0 * (3.0 * (_goal_pos - _init_pos) / ((_time_end - _time_start) * (_time_end - _time_start)) - 2.0 * _init_vel / (_time_end - _time_start) - _goal_vel / (_time_end - _time_start)) * (_time - _time_start)
			+ 3.0 * (-2.0 * (_goal_pos - _init_pos) / ((_time_end - _time_start) * (_time_end - _time_start) * (_time_end - _time_start)) + (_init_vel + _goal_vel) / ((_time_end - _time_start) * (_time_end - _time_start))) * (_time - _time_start) * (_time - _time_start);
	}

	for (int i = 0; i < _vector_size; i++) //do not use cubic spline when desired motion is small
	{
		if (abs(_goal_pos(i) - _init_pos(i)) <= _motion_threshold)
		{
			xdotd(i) = 0.0;
		}
	}

	return xdotd;
}

void CTrajectory::check_vector_size(VectorXd X)
{
	if (X.size() == _vector_size)
	{
	}
	else
	{
		cout << "Warning!!! -- Vector size in CTrajectory mismatch occured! --" << endl << endl;
	}
}

int CTrajectory::check_trajectory_complete() //1 = time when trajectory complete
{
	int diff = 0;
	bool previous_bool = _bool_trajectory_complete;
	if (_time >= _time_end && _bool_trajectory_complete == false)
	{
		_bool_trajectory_complete = true;
		diff = 1;
	}

	return diff;
}

HTrajectory::HTrajectory()
{	
	Initialize();
}

HTrajectory::~HTrajectory()
{
}

void HTrajectory::Initialize()
{
	_time_start = 0.0;
	_time = 0.0;
	_time_end = 0.0;

	XTrajectory.set_size(3);

	rotation_0.setZero();
	rotation_f.setZero();

	_init_pos.setZero(3);
	_init_vel.setZero(3);

	_goal_pos.setZero(3);
	_goal_vel.setZero(3);


	w_0.setZero();
	a_0.setZero();

	_bool_trajectory_complete = false;
}

void HTrajectory::reset_initial(double time0, VectorXd init_pos, VectorXd init_vel)
{
	if (init_pos.size() == 6 && init_vel.size() == 6)
	{
	}
	else
	{
		cout << "Warning!!! -- Vector size in HTrajectory mismatch occured! --" << endl << endl;
	}

	_time_start = time0;
	_init_pos = init_pos.head(3);
	_init_vel = init_vel.head(3);

	rotation_0 = CustomMath::GetBodyRotationMatrix(init_pos(3), init_pos(4), init_pos(5));
	XTrajectory.reset_initial(_time_start, _init_pos, _init_vel);

	_bool_trajectory_complete = false;
}

void HTrajectory::update_time(double time)
{
	_time = time;
	XTrajectory.update_time(_time);
}

void HTrajectory::update_goal(VectorXd goal_pos, VectorXd goal_vel, double goal_time)
{
	if (goal_pos.size() == 6 && goal_vel.size() == 6)
	{
	}
	else
	{
		cout << "Warning!!! -- Vector size in HTrajectory mismatch occured! --" << endl << endl;
	}

	_time_end = goal_time;
	_goal_pos = goal_pos.head(3);
	_goal_vel = goal_vel.head(3);

	rotation_f = CustomMath::GetBodyRotationMatrix(goal_pos(3), goal_pos(4), goal_pos(5));
	XTrajectory.update_goal(_goal_pos, _goal_vel, _time_end);
}

VectorXd HTrajectory::position_cubicSpline()
{
	return XTrajectory.position_cubicSpline();
}

VectorXd HTrajectory::velocity_cubicSpline()
{
	return XTrajectory.velocity_cubicSpline();
}

Matrix3d HTrajectory::rotationCubic()
{
	if (_time >= _time_end)
	{
		return rotation_f;
	}
	else if (_time < _time_start)
	{
		return rotation_0;
	}

	double tau = cubic(0, 1, 0, 0);

	Matrix3d rot_scaler_skew;
	rot_scaler_skew = (rotation_0.transpose() * rotation_f).log();
	Matrix3d result = rotation_0 * (rot_scaler_skew * tau).exp();
	return result;
}

Vector3d HTrajectory::rotationCubicDot()
{
	Matrix3d r_skew;
	r_skew = (rotation_0.transpose() * rotation_f).log();
	Vector3d a, b, c, r;
	double tau = (_time - _time_start) / (_time_end - _time_start);
	r(0) = r_skew(2, 1);
	r(1) = r_skew(0, 2);
	r(2) = r_skew(1, 0);
	c = w_0;
	b = a_0 / 2;
	a = r - b - c;
	Vector3d rd;
	for (int i = 0; i < 3; i++)
	{
		rd(i) = cubicDot(0, r(i), 0, 0);
	}
	rd = rotation_0 * rd;
	if (tau < 0) return w_0;
	if (tau > 1) return Vector3d::Zero();
	return rd; //3 * a * pow(tau, 2) + 2 * b * tau + c;
}

double HTrajectory::cubic(double x_0, double x_f, double x_dot_0, double x_dot_f)
{
	double x_t;
	if (_time < _time_start)
	{
		x_t = x_0;
	}
	else if (_time > _time_end)
	{
		x_t = x_f;
	}
	else
	{
		double elapsed_time = _time - _time_start;
		double total_time = _time_end - _time_start;
		double total_time2 = total_time * total_time;
		double total_time3 = total_time2 * total_time;
		double total_x = x_f - x_0;
		x_t = x_0 + x_dot_0 * elapsed_time
			+ (3 * total_x / total_time2
				- 2 * x_dot_0 / total_time
				- x_dot_f / total_time)
			* elapsed_time * elapsed_time
			+ (-2 * total_x / total_time3 +
			(x_dot_0 + x_dot_f) / total_time2)
			* elapsed_time * elapsed_time * elapsed_time;
	}
	return x_t;
}

double HTrajectory::cubicDot(double x_0, double x_f, double x_dot_0, double x_dot_f)
{
	double x_t;
	if (_time < _time_start)
	{
		x_t = x_dot_0;
	}
	else if (_time > _time_end)
	{
		x_t = x_dot_f;
	}
	else
	{
		double elapsed_time = _time - _time_start;
		double total_time = _time_end - _time_start;
		double total_time2 = total_time * total_time;
		double total_time3 = total_time2 * total_time;
		double total_x = x_f - x_0;

		x_t = x_dot_0
			+ 2 * (3 * total_x / total_time2
				- 2 * x_dot_0 / total_time
				- x_dot_f / total_time)
			* elapsed_time
			+ 3 * (-2 * total_x / total_time3 +
			(x_dot_0 + x_dot_f) / total_time2)
			* elapsed_time * elapsed_time;
	}
	return x_t;
}

int HTrajectory::check_trajectory_complete() //1 = time when trajectory complete
{
	int diff = 0;
	bool previous_bool = _bool_trajectory_complete;
	if (_time >= _time_end && _bool_trajectory_complete == false)
	{
		_bool_trajectory_complete = true;
		diff = 1;
	}

	return diff;
}