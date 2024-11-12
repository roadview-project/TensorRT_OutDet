#include <fstream>
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "sensor_msgs/PointCloud2.h"

int main(int argc, char **argv)
{
    // initialize ros node
    ros::init(argc, argv, "point_publisher");
    // handler for ros node
    ros::NodeHandle n;
    // topic name
    std::string topic = n.resolveName("input_cloud");
    // publisher
    ros::Publisher cloud_pub = n.advertise<sensor_msgs::PointCloud2>(topic, 10);
    // create point clouds
    sensor_msgs::PointCloud2 msg;
    //    msg.frame_id = "base_link";
    msg.is_dense = false;
    msg.is_bigendian = false;
    msg.fields.resize(4);
    msg.fields[0].name = "x";
    msg.fields[0].offset = 0;
    msg.fields[0].datatype = sensor_msgs::PointField::FLOAT32;
    msg.fields[0].count = 1;

    msg.fields[1].name = "y";
    msg.fields[1].offset = 4;
    msg.fields[1].datatype = sensor_msgs::PointField::FLOAT32;
    msg.fields[1].count = 1;

    msg.fields[2].name = "z";
    msg.fields[2].offset = 8;
    msg.fields[2].datatype = sensor_msgs::PointField::FLOAT32;
    msg.fields[2].count = 1;

    msg.fields[3].name = "i";
    msg.fields[3].offset = 12;
    msg.fields[3].datatype = sensor_msgs::PointField::FLOAT32;
    msg.fields[3].count = 1;

    msg.point_step = 16; // size of point in bytes
    // run at 10Hz
    ros::Rate loop_rate(10);

    int count = 0;
    const char* fname =  "/var/local/home/aburai/DATA/WADS2/sequences/11/velodyne/039498.bin";
    std::ifstream fin(fname, std::ios::binary);
    fin.seekg(0, std::ios::end);
    const size_t num_elements = fin.tellg() / sizeof(float);
    fin.seekg(0, std::ios::beg);
    float data[num_elements];
    fin.read(reinterpret_cast<char*>(&data[0]), num_elements * sizeof(float));
    // initial sizes
    long a_init = long(num_elements / 4);
    long b = 4;

    while (ros::ok())
    {
        // fill the message object with data
        int width = 1;
        int height = a_init;
        int num_points = width * height;
        msg.row_step = width * msg.point_step;
        msg.height = height;
        msg.width = width;
        msg.data.resize(num_points * msg.point_step);
        int pt_counter = 0;
        for (int x = 0; x < width; x++){
            for (int y=0; y< height; y++){
                uint8_t *ptr = &msg.data[0] + (x + y * width) * msg.point_step;
                *(float *)ptr = data[pt_counter];  //x
                ptr += 4;
                pt_counter += 1;
                *(float *)ptr = data[pt_counter]; //y
                ptr += 4;
                pt_counter += 1;
                *(float *)ptr = data[pt_counter]; //z
                ptr += 4;
                pt_counter += 1;
                *(float *)ptr = data[pt_counter];
                pt_counter += 1;
            }
        }
        msg.header.seq = count;
        msg.header.stamp = ros::Time::now();

        // PUBLISH  the point cloud msg
        cloud_pub.publish(msg);
        // unlike spin(), spinOnce will process the event and return so next message can be published
        ros::spinOnce();
        // time control
        loop_rate.sleep();
        ++count;
    }
    return 0;
}
