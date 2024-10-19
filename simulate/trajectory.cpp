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
	_vector_size = 1; // default = 1
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
	else
	{
		xd = _init_pos + _init_vel * (_time - _time_start) + (3.0 * (_goal_pos - _init_pos) / ((_time_end - _time_start) * (_time_end - _time_start)) - 2.0 * _init_vel / (_time_end - _time_start) - _goal_vel / (_time_end - _time_start)) * (_time - _time_start) * (_time - _time_start) + (-2.0 * (_goal_pos - _init_pos) / ((_time_end - _time_start) * (_time_end - _time_start) * (_time_end - _time_start)) + (_init_vel + _goal_vel) / ((_time_end - _time_start) * (_time_end - _time_start))) * (_time - _time_start) * (_time - _time_start) * (_time - _time_start);
	}

	for (int i = 0; i < _vector_size; i++) // do not use cubic spline when desired motion is small
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
	else
	{
		xdotd = _init_vel + 2.0 * (3.0 * (_goal_pos - _init_pos) / ((_time_end - _time_start) * (_time_end - _time_start)) - 2.0 * _init_vel / (_time_end - _time_start) - _goal_vel / (_time_end - _time_start)) * (_time - _time_start) + 3.0 * (-2.0 * (_goal_pos - _init_pos) / ((_time_end - _time_start) * (_time_end - _time_start) * (_time_end - _time_start)) + (_init_vel + _goal_vel) / ((_time_end - _time_start) * (_time_end - _time_start))) * (_time - _time_start) * (_time - _time_start);
	}

	for (int i = 0; i < _vector_size; i++) // do not use cubic spline when desired motion is small
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
	// else
	// {
	// 	cout << "Warning!!! -- Vector size in CTrajectory mismatch occured! --" << endl << endl;
	// }
}

int CTrajectory::check_trajectory_complete() // 1 = time when trajectory complete
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
		// cout << "Warning!!! -- Vector size in HTrajectory mismatch occured! --" << endl << endl;
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
		// cout << "Warning!!! -- Vector size in HTrajectory mismatch occured! --" << endl << endl;
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

// Matrix3d HTrajectory:
// 	Vector3d rot_diff_vec;
// 	Vector3d cubic_tra;
// 	Vector3d cubic_rot_tra;
// 	Matrix3d cubic_rot;
// 	m_sample.resize(12, 6);

// 	Matrix3d rot_diff = m_init.linear().inverse() * m_goal.linear() ;

// 		AngleAxisd A(rot_diff);
// 		double angle;
// 		r0 = 0.0;
// 		r1 = 0.0; //m_init.vel(i);
// 		r2 = 3.0 / pow(m_duration, 2) * (A.angle());
// 		r3 = -1.0 * 2.0 / pow(m_duration, 3) * (A.angle());

// 		angle = r0 + r1 * (m_time - m_stime) + r2 * pow(m_time - m_stime, 2) + r3 * pow(m_time - m_stime, 3);

// 		cubic_rot = m_init.linear()*AngleAngle_to_Rot(A.axis(), angle);


// Matrix3d AngleAngle_to_Rot(Vector3d axis, double angle) {

// 	Matrix3d Rot;
// 	double kx, ky, kz, theta, vt;
// 	kx = axis(0);
// 	ky = axis(1);
// 	kz = axis(2);
// 	theta = angle;
// 	vt = 1.0 - cos(theta);


// 	Rot(0, 0) = kx * kx*vt + cos(theta);
// 	Rot(0, 1) = kx * ky*vt - kz * sin(theta);
// 	Rot(0, 2) = kx * kz*vt + ky * sin(theta);
// 	Rot(1, 0) = kx * ky*vt + kz * sin(theta);
// 	Rot(1, 1) = ky * ky*vt + cos(theta);
// 	Rot(1, 2) = ky * kz*vt - kx * sin(theta);
// 	Rot(2, 0) = kx * kz*vt - ky * sin(theta);
// 	Rot(2, 1) = ky * kz*vt + kx * sin(theta);
// 	Rot(2, 2) = kz * kz*vt + cos(theta);

// 	return Rot;

// }
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
	if (tau < 0)
		return w_0;
	if (tau > 1)
		return Vector3d::Zero();
	return rd; // 3 * a * pow(tau, 2) + 2 * b * tau + c;
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
		x_t = x_0 + x_dot_0 * elapsed_time + (3 * total_x / total_time2 - 2 * x_dot_0 / total_time - x_dot_f / total_time) * elapsed_time * elapsed_time + (-2 * total_x / total_time3 + (x_dot_0 + x_dot_f) / total_time2) * elapsed_time * elapsed_time * elapsed_time;
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

		x_t = x_dot_0 + 2 * (3 * total_x / total_time2 - 2 * x_dot_0 / total_time - x_dot_f / total_time) * elapsed_time + 3 * (-2 * total_x / total_time3 + (x_dot_0 + x_dot_f) / total_time2) * elapsed_time * elapsed_time;
	}
	return x_t;
}

int HTrajectory::check_trajectory_complete() // 1 = time when trajectory complete
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

////////////////////////////////////////////////////////////////
/* pos, vel 3개 짜리 */
RTrajectory::RTrajectory()
{
	Initialize();
}

RTrajectory::~RTrajectory()
{
}

void RTrajectory::Initialize()
{
	_time_start = 0.0;
	_time = 0.0;
	_time_end = 0.0;

	_theta = 0;
	_dtheta = 0;
	_goal_theta = 0;
	_rpy_next.setZero(3);
	_tangent_vector.setZero(3);
	_normal_vector.setZero(3);
	_Tvr.setIdentity(4,4);

	_bool_trajectory_complete = false;
}
void RTrajectory::reset_initial(double time0, Vector3d tangent_vector, Vector3d normal_vector, Vector3d origin, double radius, MatrixXd Tvr, double dt)
{
	
	_time_start = time0;
	_time = _time_start;
	_tangent_vector = tangent_vector;
	_normal_vector = normal_vector;
	_r = radius;
	_Tvr = Tvr;
	_dt = dt;
	_acc_theta = DEG2RAD * 10; //[degree]
	_acc_time = 0.2;
	_add_theta = 0;
	_origin = origin;
	

	_bool_trajectory_complete = false;

	_theta_next = angle_cubicSpline();
	_dtheta_next = angledot_cubicSpline();
	
}



double RTrajectory::update_time(double time)
{
	_time = time+_dt;

	_theta = _theta_next;
	
	_dtheta = _dtheta_next;
	_theta_next = angle_cubicSpline();
	_dtheta_next = angledot_cubicSpline();
	return _theta_next;
}

void RTrajectory::update_theta(double theta)
{
	// _theta = _theta_next;
	// _theta_next = theta;
	// _dtheta = _dtheta_next;
	// _dtheta_next = (_theta_next - _theta) / _dt;
	_theta_next = _theta; //일단 theta next를 previous one 으로 상정하고...
	_theta = theta;
	_dtheta = (_theta - _theta_next) / _dt;
}

void RTrajectory::update_goal(double goal_time, double init_theta, double goal_theta, Vector3d x_hand)
{	
	_add_theta = init_theta;
	_init_theta = 0;
	_goal_theta = goal_theta-init_theta;
	_time_end = goal_time;

	if (goal_theta > init_theta){
		_acc_theta = 1 * _acc_theta + _init_theta;
	}
	else{
		_acc_theta = -1* _acc_theta + _init_theta;
	}
	_acc_dtheta = (_goal_theta - _acc_theta - (_init_theta+_acc_theta)) / (_time_end - (_acc_time) - (_acc_time + _time_start));
	
	_rpy_next = rotation_circular_calc(init_theta, 0., 0., x_hand);
	// cout<<"####\n\nupdate goal\n\n###\n";

	_theta_next = init_theta;
	_theta = init_theta;
	
}

VectorXd RTrajectory::position_circular()
{
	VectorXd xd(3);
	VectorXd tmp(4);
	tmp << _r * cos(_theta), _r * sin(_theta), 0, 1;
	tmp << _Tvr * tmp;

	xd << tmp.head(3);

	// std::cout<<"theta in trajectory :"<<_theta<<std::endl;

	return xd;
}

VectorXd RTrajectory::velocity_circular()
{
	VectorXd xdotd(3);
	VectorXd tmp(4);
	tmp << _r*_dtheta*-sin(_theta), _r*_dtheta*cos(_theta), 0 , 0;
	tmp << _Tvr*tmp;
	xdotd<<tmp.head(3);
	// cout<<"@@\n\n"<<_theta<<endl<<_Tvr<<endl<<_r<<endl<<_dtheta<<"\n\n@@"<<endl;
	
	return xdotd;
}


double RTrajectory::angle_cubicSpline()
{
	double th1, th2, dth1, dth2, t_t0, theta;
	
	if (_time <= _time_start)
	{
		theta = _init_theta;
	}

	else if ((_time_start < _time) && (_time <= _acc_time + _time_start))
	{
		th1 = _init_theta;
		th2 = _init_theta+_acc_theta;
		dth1 = 0;
		dth2 = _acc_dtheta;
		t_t0 = _time - _time_start;
		theta = th1 + dth1 * t_t0 + (3 * (th2 - th1) - _acc_time * (2 * dth1 + dth2)) / pow(_acc_time, 2) * pow(t_t0, 2) + (-2 * (th2 - th1) + _acc_time * (dth1 + dth2)) / pow(_acc_time, 3) * pow(t_t0, 3);
		
	}
	else if ((_acc_time + _time_start < _time) && (_time <= (_time_end - (_acc_time))))
	{
		theta = _acc_dtheta*(_time - (_acc_time + _time_start)) + _acc_theta;
		
	}
	else if (((_time_end - (_acc_time)) < _time) && (_time <= _time_end))
	{
		th1 = _goal_theta - _acc_theta;
		th2 = _goal_theta;
		dth1 = _acc_dtheta;
		dth2 = 0;
		t_t0 = _time - (_time_end - (_acc_time));
		theta = th1 + dth1 * t_t0 + (3 * (th2 - th1) - _acc_time * (2 * dth1 + dth2)) / pow(_acc_time, 2) * pow(t_t0, 2) + (-2 * (th2 - th1) + _acc_time * (dth1 + dth2)) / pow(_acc_time, 3) * pow(t_t0, 3);
		
	}
	else if (_time > _time_end)
	{
		theta = _goal_theta;
	}
	return theta+_add_theta;
}

double RTrajectory::angledot_cubicSpline()
{
	double th1, th2, dth1, dth2, t_t0, dtheta;

	if (_time <= _time_start)
	{
		dtheta = 0;
	}

	else if ((_time_start < _time) && (_time <= _acc_time + _time_start))
	{
		th1 = _init_theta;
		th2 = _init_theta+_acc_theta;
		dth1 = 0;
		dth2 = _acc_dtheta;
		t_t0 = _time - _time_start;
		dtheta = dth1 + 2 * (3 * (th2 - th1) - _acc_time * (2 * dth1 + dth2)) / pow(_acc_time, 2) * t_t0 + 3 * (-2 * (th2 - th1) + _acc_time * (dth1 + dth2)) / pow(_acc_time, 3) * pow(t_t0, 2);

	
	}
	else if ((_acc_time + _time_start < _time) && (_time <= (_time_end - (_acc_time))))
	{
		dtheta = _acc_dtheta;
		
	}
	else if (((_time_end - (_acc_time)) < _time) && (_time <= _time_end))
	{
		th1 = _goal_theta - _acc_theta;
		th2 = _goal_theta;
		dth1 = _acc_dtheta;
		dth2 = 0;
		t_t0 = _time - (_time_end - (_acc_time));
		dtheta = dth1 + 2 * (3 * (th2 - th1) - _acc_time * (2 * dth1 + dth2)) / pow(_acc_time, 2) * t_t0 + 3 * (-2 * (th2 - th1) + _acc_time * (dth1 + dth2)) / pow(_acc_time, 3) * pow(t_t0, 2);
		
	}
	else if (_time > _time_end)
	{
		dtheta = 0;
	}
	return dtheta;
}

Matrix3d RTrajectory::R3D(Vector3d unitVec, double angle)
{
	Matrix3d _Tug;
	double cosAngle = cos(angle);
	double sinAngle = sin(angle);
	double x = unitVec(0);
	double y = unitVec(1);
	double z = unitVec(2);
	Matrix3d rotMatrix;
	rotMatrix << cosAngle + (1 - cosAngle) * x * x, (1 - cosAngle) * x * y - sinAngle * z, (1 - cosAngle) * x * z + sinAngle * y,
		(1 - cosAngle) * y * x + sinAngle * z, cosAngle + (1 - cosAngle) * y * y, (1 - cosAngle) * y * z - sinAngle * x,
		(1 - cosAngle) * z * x - sinAngle * y, (1 - cosAngle) * z * y + sinAngle * x, cosAngle + (1 - cosAngle) * z * z;

	_Tug << rotMatrix * _tangent_vector,
		rotMatrix *_normal_vector.cross(_tangent_vector),
		rotMatrix * _normal_vector;
	
	return _Tug;
}

Vector3d RTrajectory::rotation_circular_calc(double theta, double pi, double psi, Vector3d x_hand){
	
	Matrix3d Tug, Tge, Tue;
	Matrix3d Troll, Tpitch, Troll_inv, Tpitch_inv;
	double angle;
	double ee_align = DEG2RAD*(45); // end effector와 gripper 가 yaw방향으로 45도 틀어져있음 
	angle = -theta; // frame은 반대 방향으로 회전 해야지, gripper방향이 유지된다.
	Vector3d x_direction, y_direction;
    double roll, pitch, yaw;
	Vector3d rpy;
    
	Tge << cos(ee_align), -sin(ee_align), 0,
			sin(ee_align), cos(ee_align), 0,
			0, 0, 1;

	Tug = R3D(_normal_vector, angle);
	
	MatrixXd _Tug_normal,_Tug_y;
	MatrixXd _Tug_normal_inv,_Tug_y_inv;
	_Tug_normal.setZero(4,4);
	_Tug_normal_inv.setZero(4,4);
	_Tug_y.setZero(4,4);
	_Tug_y_inv.setZero(4,4);
	

	
	// x_direction << obj.r_margin.normalized();
	x_direction << (x_hand - _origin).normalized();
	y_direction << _normal_vector.cross(x_direction);
	Troll = R3D(x_direction, psi);
	Troll_inv =R3D(x_direction, 0);
	Tpitch = R3D(y_direction, pi);
	Tpitch_inv = R3D(y_direction, 0);

	// Tue << Troll*Troll.inverse()*Tpitch*Tpitch_inv.inverse()*Tug*Tge; 
	Tue << Troll*Troll_inv.inverse()*Tpitch*Tpitch_inv.inverse()*Tug*Tge; 

	// Matrix3d Rot_x, Rot_y;
	// Rot_x.setZero(3,3);Rot_y.setZero(3,3);
	// Rot_x << 1, 0, 0,
	// 		0, cos(psi), -sin(psi),
	// 		0, sin(psi), cos(psi);
			
	// Rot_y << cos(pi),0, -sin(pi),
	// 		0 , 1, 0,
	// 		sin(pi),0 ,cos(pi);
	
	// Tue = Tug*Rot_x*Rot_y*Tge; // 이렇게 계산하면 roll이랑 Pitch direction이 반대로 돌아간당.. 

	yaw = atan2(Tue(1,0),Tue(0,0));
	pitch = atan2(-Tue(2,0), sqrt(pow(Tue(2,1),2) + pow(Tue(2,2),2)));
	roll = atan2(Tue(2,1),Tue(2,2));
	
	
	yaw = fmod(yaw + M_PI, 2 * M_PI);
    if (yaw < 0) {
        yaw += 2 * M_PI;
    }
    yaw = yaw - M_PI;

	pitch = fmod(pitch + M_PI, 2 * M_PI);
    if (pitch < 0) {
        pitch += 2 * M_PI;
    }
    pitch = pitch - M_PI;

	roll = fmod(roll + M_PI, 2 * M_PI);
	
	if (roll < 0) {
        roll += 2 * M_PI;
    }
	roll = roll - M_PI;

	rpy<<roll,pitch,yaw;	
	
	// cout<<"rpy :"<<rpy.transpose()<<"  xhand:"<<x_hand.transpose()<<endl;
	return rpy;


}

Vector3d RTrajectory::rotation_circular(double pi, double psi,  Vector3d x_hand){
	_rpy = _rpy_next;
	// _rpy_next = rotation_circular_calc(_theta_next,-DEG2RAD*10,0, x_hand);
	_rpy_next = rotation_circular_calc(_theta_next, pi, psi, x_hand);
	
	return _rpy;

}
Vector3d RTrajectory::rotationdot_circular(){
	Matrix3d r0 = CustomMath::GetBodyRotationMatrix(_rpy(0),_rpy(1),_rpy(2));
	Matrix3d r1 = CustomMath::GetBodyRotationMatrix(_rpy_next(0),_rpy_next(1),_rpy_next(2));
	Matrix3d rdiff = r1 * r0.transpose();
	Matrix3d r_skew = rdiff.log();
	Vector3d drpy;
	drpy(0) = r_skew(2, 1);
	drpy(1) = r_skew(0, 2);
	drpy(2) = r_skew(1, 0);
	
	return drpy/_dt;
}

//dependent function. 
Vector3d RTrajectory::drpy2nextrpy(Vector3d drpy, Vector3d rpy){
	Matrix3d r_skew;
	// r_skew << 0,     -drpy(2),  drpy(1),
	// 		 drpy(2), 0,       -drpy(0),
	// 		-drpy(1), drpy(0),	 0;
	r_skew << 0,     -drpy[2]*_dt,  drpy[1]*_dt,
			 drpy[2]*_dt, 0,       -drpy[0]*_dt,
			-drpy[1]*_dt, drpy[0]*_dt,	 0;
	// r_skew << 0,     -drpy[2],  drpy[1],
	// 		 drpy[2], 0,       -drpy[0],
	// 		-drpy[1], drpy[0],	 0;
	Matrix3d rdiff = r_skew.exp();
	Matrix3d r0 = CustomMath::GetBodyRotationMatrix(rpy(0),rpy(1),rpy(2));
	Matrix3d r1 = rdiff * r0.transpose().inverse();

	Vector3d rpy_next = CustomMath::GetBodyRotationAngle(r1);
	return rpy_next;
	
}

int RTrajectory::check_trajectory_complete() // 1 = time when trajectory complete
{
	int diff = 0;
	// cout<<"st :"<<_time_start<<" ed:"<<_time_end<<" time :"<<_time<<endl;
	bool previous_bool = _bool_trajectory_complete;
	if (_time >= _time_end && previous_bool == false)
	{
		_bool_trajectory_complete = true;
		diff = 1;
	}
	return diff;
}