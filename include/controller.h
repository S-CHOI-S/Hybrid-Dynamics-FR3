#pragma once
#ifndef __CONTROLLER_H
#define __CONTROLLER_H

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <rbdl/rbdl.h>
#include <rbdl/addons/urdfreader/urdfreader.h>
#include <vector>

#include "robotmodel.h"
#include "trajectory.h"
#include "custommath.h"

using namespace std;
using namespace Eigen;

#define NECS2SEC 1000000000

class CController
{

public:
    CController(int JDOF);
    virtual ~CController();	

    void control_mujoco();
    

    // void read_pybind(double t, std::array<double,9> q, std::array<double, 9> qdot, double timestep,std::array<double, 66> pos);
    void read_pybind(double t, array<double, 9> q, array<double, 9> qdot, double timestep);
 
    tuple<std::vector<double>, double> write_pybind();
    double write_force_pybind();
    void put_action_pybind(array<double, 7> torque);
    
    tuple<array<double,2>, double> get_actions_pybind();
    // tuple<array<double,2>, double> get_commands_pybind();
    array<double, 6> get_commands_pybind();
    // void put_action_pybind(std::array<double, 2> action);

    // void put_action2_pybind(std::array<double, 2> action);
    // void put_action3_pybind(std::array<double, 3> action);
    
    // void put_action2_pybind(std::array<double, 3> drpy,std::array<double, 3> dxyz, double gripper);
    // void put_action3_pybind(std::array<double, 5> compensate_force);
    
    void randomize_env_pybind(std::array<std::array<double, 3>, 3> rotation_obj, std::string object_name, int scale, std::array<double, 66> pos, double init_theta, double goal_theta, int planning_mode, bool generate_dxyz);
    tuple<std::vector<double>, std::vector<double>> get_force_pybind();
    double control_mode_pybind();
    double write_control_mode_pybind(double ctrl_mode);

    // tuple<vector<double>,vector<double>, vector<vector<double>>> get_ee_pybind();
    tuple<vector<double>, vector<double>, float, float, float, float> get_ee_pybind();
    vector<vector<double>> get_jacobian_pybind();
    tuple<vector<vector<double>>,vector<vector<double>>,vector<vector<double>>> get_model_pybind();
    vector<double> desired_rpy_pybind();
    array<double, 16> relative_T_hand_pybind();
    void TargetRePlan_pybind();
    void TargetRePlan2_pybind(std::array<double, 7> q_goal);
    void TargetRePlan3_pybind(std::array<double, 7> q_goal);
    double get_force_gain();

    // double TargetPlanRL_pybind(double angle);
    // tuple<double,vector<double>,vector<double>,vector<double>>TargetPlanRL_pybind(double angle);
    std::vector<double> torque_command, force_command;
    vector<double> x_hand;
	vector<double> x_plan;
	vector<double> xdot_hand;
	vector<double> rpy_des;
    float gripper_goal;
	vector<vector<double>> J_hands, lambda, inertia;
	
    
    void Initialize();
    void Initialize_Cabinet(int planning_mode, array<array<double, 4>, 4> cabinet_info, double goal_theta);
    


private:
    
    void ModelUpdate();
    void motionPlan();
    void motionPlan_taskonly();
    void motionPlan_Heuristic(const char* object, double init_theta, double goal_theta);
    
    void motionPlan_RL(string object);
    double lpf(double input, double previousOutput, double alpha);
    Matrix3d rotation_calc(double pi, double psi, Matrix3d R_hand);
    Vector3d rotationdot_circular(Matrix3d r1, Matrix3d r0);
    struct Robot{
		int id;
		Vector3d pos;
		double zrot; //frame rotation according to Z-axis
		double ee_align; //gripper and body align angle

	};

	struct Objects{
        const char* name;
		int id;
		Vector3d o_margin; // (x,y,z)residual from frame origin to rotation plane
		Vector3d r_margin; // (x,y,z)radius of the object (where we first grab)
		Vector3d grab_dir; // grabbing direction. r_margin vs o_margin x r_margin
        Vector3d norm_dir;
        Vector3d pos;
        Vector3d origin;
	};

    struct Target{
        double x;
        double y;
        double z;
        double roll;
        double pitch;
        double yaw;
        double gripper;
        double time;
        Vector3d target_velocity;
        VectorXd q_goal;
        string state;
    };


    void reset_target(double motion_time, VectorXd target_joint_position);
    void reset_target(double motion_time, VectorXd target_joint_position, VectorXd target_joint_velocity);
    void reset_target(double motion_time, Vector3d target_pos, Vector3d target_ori);
    void reset_target(double motion_time, VectorXd target_joint_position, string state);
    
    //task space control with target velocity 
    void reset_target(double motion_time, Vector3d target_pos, Vector3d target_ori, Vector3d target_velocity);
    

    //reset target for circular trajectory 
    void reset_target(double motion_time, string state); 
    void reset_target(double motion_time, Vector3d target_pos, Vector3d target_ori, string state);
    
    void load_model();
    VectorXd _q; // joint angle
	VectorXd _qdot; // joint velocity
    VectorXd _torque; // joint torque

    // VectorXd _gripper; // gripper joint angle
	// VectorXd _gripperdot; // gripper joint velocity
    // VectorXd _grippertorque; // gripper joint torque
    double _gripper; // gripper joint angle
	double _gripperdot; // gripper joint velocity
    double _grippertorque; // gripper joint torque

    VectorXd _valve; // position of the valve
    VectorXd _handle_valve; // positon of the handle valve
    VectorXd _robot_base; // postion of the robot base .... we consider the rotation of the robot and the objects are fixed (temporarily) 
    MatrixXd _rotation_obj; // rotation from environment randomization 
    int _k; // DOF

    bool _bool_init;
    double _t;
    double _dt;
	double _init_t;
	double _pre_t;
    VectorXd q_goal;
    double du;

    int _planning_mode; //0 : heuristic, 1 : heuristic object random, 2: RL object random

    //controller
	double _kpj, _kdj; //joint P,D gain
    double _kpj_gripper, _kdj_gripper; //gripper P,D gain
    double _x_kp; // task control P gain

    void JointControl();
    void GripperControl();
    void CLIK();
    void OperationalSpaceControl();
    void HybridControl();

    // robotmodel
    CModel Model;

    int _cnt_plan;
	VectorXd _time_plan;
	VectorXi _bool_plan;

    int _control_mode; //1: joint space, 2: operational space
    int _gripper_mode; //0: close, 2: open
    bool _init_mp;
    VectorXd _q_home; // joint home position

    //motion trajectory
	double _start_time, _end_time, _motion_time;

    CTrajectory JointTrajectory; // joint space trajectory
    HTrajectory HandTrajectory; // task space trajectory
    RTrajectory CircularTrajectory; 
    RTrajectory CircularTrajectory2; 
    

    bool _bool_joint_motion, _bool_ee_motion; // motion check

    // VectorXd _q_des, _qdot_des, _gripper_des, _q_pre, _qdot_pre; 
    // VectorXd _q_goal, _qdot_goal, _gripper_goal, _gripperdot_goal, _init_gripper;
    VectorXd _q_des, _qdot_des, _q_pre, _qdot_pre; 
    VectorXd _q_goal, _qdot_goal;
    VectorXd _x_des_hand, _xdot_des_hand;
    VectorXd _x_goal_hand, _xdot_goal_hand;
    Vector3d _pos_goal_hand, _rpy_goal_hand;
    double _gripper_des, _gripper_goal, _gripperdot_goal, _init_gripper;

    MatrixXd _A_diagonal; // diagonal inertia matrix
    MatrixXd _J_hands; // jacobian matrix
    MatrixXd _J_bar_hands; // pseudo invere jacobian matrix

    VectorXd _x_hand, _xdot_hand; // End-effector

    VectorXd _x_tmp;
    VectorXd _x_err_hand, _xdot_err_hand;
    Matrix3d _R_des_hand, _R_hand, _R_hand_ref;
    Matrix3d _Rdot_des_hand, _Rdot_hand;
    MatrixXd _lambda;
    VectorXd _force, _compensate_force, _compensated_force;
    VectorXd _qdot_rl;

    MatrixXd _I; // Identity matrix


    vector<Target> _target_plan;
    
    CController::Target TargetTransformMatrix(Objects obj, Robot robot, double angle);
    CController::Target rpyTransformMatrix(Objects obj, Robot robot, double angle);
    Matrix3d R3D(Objects obj, Vector3d unitVec, double angle);
    // void TargetPlanHeuristic1(Objects obj, Robot robot, double angle);
    // void TargetPlanHeuristic2(Objects obj, Robot robot, double init_theta, double goal_theta);
    void TargetPlan();
    // void TargetPlanRL2();
    // Vector3d AddTorque(Objects _obj, Vector3d _x_hand, Matrix4d _Tbu, Matrix4d _Tvb);
    Vector3d AddTorque();
    void writeToCSVfile(string filename, MatrixXd input_matrix, int cnt);

    VectorXd _grab_vector, _normal_vector, _origin;
    Matrix4d _Tvr, _Tvb, _Tbu, _Tur;
    MatrixXd _Rvu_J, _Ruv_J;
    Matrix3d _Rvu, _Ruv;
    
    double _radius, _init_theta, _goal_theta;
    Objects _obj;
    Robot _robot;

    vector<double> dr;
    vector<double> dp;
    vector<double> dy;
    int Ccount;

	vector<double> _x_plan;
	vector<double> _y_plan;
	vector<double> _z_plan;
    vector<double> compensate_force;
    
    double _theta_des;
    VectorXd accum_err;
    string _object_name;
    VectorXd _accum_err_x, _accum_err_q;
    MatrixXd _kpj_diagonal, _kdj_diagonal;
    MatrixXd _kpj_diagonal6, _kdj_diagonal6;
    MatrixXd _R_fix;
    VectorXd _drpy, _rpy_des, _dxyz;
    VectorXd _x_force;
    VectorXd _Force;
    vector<VectorXd> _q_goal_data;
    double _gripper_close, _gripper_open;
    bool _generate_dxyz;
    double _print_time, _print_interval;
    double _scale_obj;
    int scale;

    VectorXd _x_circular_hand;
    vector<double> x_circular_hand;

    Target target;
    int _cnt_plot;
    MatrixXd _q_plot, _qdot_plot, _x_plot;

    MatrixXd _Null;
    VectorXd _q_mid, _qnull_err;
    MatrixXd _eye;
    Vector3d _v2ei_vector, _v2ef_vector;

    double _dtheta, _dtheta_des;
    double _force_gain, _dforce_gain, _rforce_gain;
    double _roll, _pitch, _droll, _dpitch, _ddroll, _ddpitch;
    double _kf;
    double _q_valve, _qdot_valve;

    ////////////////
    double _q_door, _q_latch, _qdot_door, _qdot_latch, _q_cabinet, _qdot_cabinet;
    Matrix3d _rotation_latch, _rotation_door;
    Vector3d _position_latch, _position_door;
    Objects _latch, _door, _cabinet;
    double _goal_theta_latch, _goal_theta_door, _goal_theta_cabinet;
    
    // for cabinet
    Matrix3d _rotation_cabinet;
    Vector3d _position_cabinet;

    VectorXd action_torque;

};

#endif


// #pragma once
// #ifndef __CONTROLLER_H
// #define __CONTROLLER_H

// #include <iostream>
// #include <eigen3/Eigen/Dense>

// using namespace std;
// using namespace Eigen;

// #define PI 3.141592
// #define DEG2RAD (0.01745329251994329576923690768489)
// #define RAD2DEG 1.0/DEG2RAD

// class Controller
// {
// public:
//     Controller(int JDOF);
// 	virtual ~Controller();	
//     void read(double time, double* q, double* qdot, double timestep);
//     void write(double* torque);

// public:
//     VectorXd _torque; //7
//     VectorXd _q; //joint angle vector
// 	VectorXd _qdot; //joint velocity vector

//     int _dt;
//     int _t;
// private:
//     void Initialize();
//     int _dofj;
//     int _kp;
//     int _kd;


//     void outputAsMatrix(const Eigen::Quaterniond& q);
//     VectorXd _qdes;
//     VectorXd _qdot_des;

//     void JointControl();
// };

// #endif
