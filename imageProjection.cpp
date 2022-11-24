// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following papers:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.
//   T. Shan and B. Englot. LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry and Mapping on Variable Terrain
//      IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). October 2018.

#include "utility.h"

class ImageProjection{
private:

    ros::NodeHandle nh; //创建ros句柄

    ros::Subscriber subLaserCloud; //订阅点云
     
    ros::Publisher pubFullCloud; //发布全部点云
    ros::Publisher pubFullInfoCloud; //发布全部点云信息

    ros::Publisher pubGroundCloud; //发布地面点云
    ros::Publisher pubSegmentedCloud; //发布被分割的点云
    ros::Publisher pubSegmentedCloudPure;
    ros::Publisher pubSegmentedCloudInfo;
    ros::Publisher pubOutlierCloud;

    pcl::PointCloud<PointType>::Ptr laserCloudIn;
    pcl::PointCloud<PointXYZIR>::Ptr laserCloudInRing;

    pcl::PointCloud<PointType>::Ptr fullCloud; // projected velodyne raw cloud, but saved in the form of 1-D matrix
    pcl::PointCloud<PointType>::Ptr fullInfoCloud; // same as fullCloud, but with intensity - range

    pcl::PointCloud<PointType>::Ptr groundCloud;
    pcl::PointCloud<PointType>::Ptr segmentedCloud;
    pcl::PointCloud<PointType>::Ptr segmentedCloudPure;
    pcl::PointCloud<PointType>::Ptr outlierCloud;

    PointType nanPoint; // fill in fullCloud at each iteration

    cv::Mat rangeMat; // range matrix for range image
    cv::Mat labelMat; // label matrix for segmentaiton marking
    cv::Mat groundMat; // ground matrix for ground cloud marking
    int labelCount;

    float startOrientation;
    float endOrientation;

    cloud_msgs::cloud_info segMsg; // info of segmented cloud
    std_msgs::Header cloudHeader;

    std::vector<std::pair<int8_t, int8_t> > neighborIterator; // neighbor iterator for segmentaiton process

    uint16_t *allPushedIndX; // array for tracking points of a segmented object
    uint16_t *allPushedIndY;

    uint16_t *queueIndX; // array for breadth-first search process of segmentation, for speed
    uint16_t *queueIndY;

public:
    ImageProjection():
        nh("~"){
        //订阅激光雷达点云信息（订阅消息话题，队列，回调函数,this?）
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 1, &ImageProjection::cloudHandler, this);

        pubFullCloud = nh.advertise<sensor_msgs::PointCloud2> ("/full_cloud_projected", 1);
        pubFullInfoCloud = nh.advertise<sensor_msgs::PointCloud2> ("/full_cloud_info", 1);

        pubGroundCloud = nh.advertise<sensor_msgs::PointCloud2> ("/ground_cloud", 1);
        pubSegmentedCloud = nh.advertise<sensor_msgs::PointCloud2> ("/segmented_cloud", 1);
        pubSegmentedCloudPure = nh.advertise<sensor_msgs::PointCloud2> ("/segmented_cloud_pure", 1);
        pubSegmentedCloudInfo = nh.advertise<cloud_msgs::cloud_info> ("/segmented_cloud_info", 1);
        pubOutlierCloud = nh.advertise<sensor_msgs::PointCloud2> ("/outlier_cloud", 1);

        nanPoint.x = std::numeric_limits<float>::quiet_NaN();
        nanPoint.y = std::numeric_limits<float>::quiet_NaN();
        nanPoint.z = std::numeric_limits<float>::quiet_NaN();
        nanPoint.intensity = -1;
        //申请资源
        allocateMemory();
        //初始化
        resetParameters();
    }

    void allocateMemory(){

        laserCloudIn.reset(new pcl::PointCloud<PointType>());
        laserCloudInRing.reset(new pcl::PointCloud<PointXYZIR>());

        fullCloud.reset(new pcl::PointCloud<PointType>());
        fullInfoCloud.reset(new pcl::PointCloud<PointType>());

        groundCloud.reset(new pcl::PointCloud<PointType>());
        segmentedCloud.reset(new pcl::PointCloud<PointType>());
        segmentedCloudPure.reset(new pcl::PointCloud<PointType>());
        outlierCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);
        fullInfoCloud->points.resize(N_SCAN*Horizon_SCAN);

        segMsg.startRingIndex.assign(N_SCAN, 0);
        segMsg.endRingIndex.assign(N_SCAN, 0);

        segMsg.segmentedCloudGroundFlag.assign(N_SCAN*Horizon_SCAN, false);
        segMsg.segmentedCloudColInd.assign(N_SCAN*Horizon_SCAN, 0);
        segMsg.segmentedCloudRange.assign(N_SCAN*Horizon_SCAN, 0);

        std::pair<int8_t, int8_t> neighbor;
        neighbor.first = -1; neighbor.second =  0; neighborIterator.push_back(neighbor);
        neighbor.first =  0; neighbor.second =  1; neighborIterator.push_back(neighbor);
        neighbor.first =  0; neighbor.second = -1; neighborIterator.push_back(neighbor);
        neighbor.first =  1; neighbor.second =  0; neighborIterator.push_back(neighbor);

        allPushedIndX = new uint16_t[N_SCAN*Horizon_SCAN];
        allPushedIndY = new uint16_t[N_SCAN*Horizon_SCAN];

        queueIndX = new uint16_t[N_SCAN*Horizon_SCAN];
        queueIndY = new uint16_t[N_SCAN*Horizon_SCAN];
    }

    void resetParameters(){
        laserCloudIn->clear();
        groundCloud->clear();
        segmentedCloud->clear();
        segmentedCloudPure->clear();
        outlierCloud->clear();

        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
        groundMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0));
        labelMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(0));
        labelCount = 1;

        std::fill(fullCloud->points.begin(), fullCloud->points.end(), nanPoint);
        std::fill(fullInfoCloud->points.begin(), fullInfoCloud->points.end(), nanPoint);
    }

    ~ImageProjection(){}

    void copyPointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){

        cloudHeader = laserCloudMsg->header;
        // cloudHeader.stamp = ros::Time::now(); // Ouster lidar users may need to uncomment this line
        pcl::fromROSMsg(*laserCloudMsg, *laserCloudIn);
        // Remove Nan points
        std::vector<int> indices; //indices记录的是去除无效点的点云索引（比如第三个点无效，则indices中的索引值为1，2，4，5...）
        pcl::removeNaNFromPointCloud(*laserCloudIn, *laserCloudIn, indices);
        // have "ring" channel in the cloud
        if (useCloudRing == true){
            pcl::fromROSMsg(*laserCloudMsg, *laserCloudInRing);
            if (laserCloudInRing->is_dense == false) {
                ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
                ros::shutdown();
            }  
        }
    }
    
    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){

        // 1. Convert ros message to pcl point cloud
        copyPointCloud(laserCloudMsg);
        // 2. Start and end angle of a scan
        findStartEndAngle();
        // 3. Range image projection
        projectPointCloud();
        // 4. Mark ground points
        groundRemoval();
        // 5. Point cloud segmentation
        cloudSegmentation();
        // 6. Publish all clouds
        publishCloud();
        // 7. Reset parameters for next iteration
        resetParameters();
    }
    // 找到一帧中起始点和终止点的角度
    void findStartEndAngle(){
        // start and end orientation of this cloud
        // 计算水平视角上该帧开始的角度
        segMsg.startOrientation = -atan2(laserCloudIn->points[0].y, laserCloudIn->points[0].x);
        // 计算水平视角上该帧结束的角度 一帧Lidar点云数据的扫描范围应该等于2*PI
        segMsg.endOrientation   = -atan2(laserCloudIn->points[laserCloudIn->points.size() - 1].y,
                                                     laserCloudIn->points[laserCloudIn->points.size() - 1].x) + 2 * M_PI;
        if (segMsg.endOrientation - segMsg.startOrientation > 3 * M_PI) {
            segMsg.endOrientation -= 2 * M_PI;
        } else if (segMsg.endOrientation - segMsg.startOrientation < M_PI)
            segMsg.endOrientation += 2 * M_PI;
        segMsg.orientationDiff = segMsg.endOrientation - segMsg.startOrientation;
        }

    void projectPointCloud(){
        // range image projection
        float verticalAngle, horizonAngle, range;
        size_t rowIdn, columnIdn, index, cloudSize; 
        PointType thisPoint;

        cloudSize = laserCloudIn->points.size();
        //遍历点云
        for (size_t i = 0; i < cloudSize; ++i){

            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            // find the row and column index in the iamge for this point
            //如果有ring indix则直接使用
            if (useCloudRing == true){
                rowIdn = laserCloudInRing->points[i].ring;
            }
            //否则通过计算垂直角度，并根据安装角度和垂直分辨率，判断该点属于哪一行
            else{
                verticalAngle = atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;
                rowIdn = (verticalAngle + ang_bottom) / ang_res_y;
            }
            //若该行的信息小于0或者大于激光雷达的线数，即跳过该点。
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;
            //计算该点的水平角度
            horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
            //计算该点所在列
            columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;

            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;
            //计算该点到原点的距离
            range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);
            //如果距离小于1m，则过滤。通常用来过滤自身形成的点云
            if (range < sensorMinimumRange)
                continue;
            //保存距离信息到矩阵对应位置即（rowIdn,cloumnIdn）
            rangeMat.at<float>(rowIdn, columnIdn) = range;
            //这里的强度实际上是保存纵坐标和横坐标
            thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;
            //把点云保存到数组
            //index为该点在点云中的索引
            index = columnIdn  + rowIdn * Horizon_SCAN;
            //fullCloud的强度信息为点云点的横纵坐标
            fullCloud->points[index] = thisPoint;
            //fullInfoCloud的强度信息为点云点的距离信息
            fullInfoCloud->points[index] = thisPoint;
            fullInfoCloud->points[index].intensity = range; // the corresponding range of a point is saved as "intensity"
        }
    }

    /*
        地面检测原理：
            通过判断rangeImage中两个相邻点之间的夹角ɑ是否足够小，
            如果角度足够小则判断为地面。这里选择的阈值为10°，小于10°则为地面
    */
    void groundRemoval(){
        size_t lowerInd, upperInd;
        float diffX, diffY, diffZ, angle;
        // groundMat
        // -1, no valid info to check if ground of not 无效地面点
        //  0, initial value, after validation, means not ground 初始点
        //  1, ground 有效地面点

        // 遍历激光束小于7的点云点（Lidar发出的激光束从最下面一条向上依次是0，1，2...）
        // 也就是遍历了7*1800个点云点,判断是否为地面点，并打上标签。
        for (size_t j = 0; j < Horizon_SCAN; ++j){
            for (size_t i = 0; i < groundScanInd; ++i){
                // 取出一个点云点 a
                lowerInd = j + ( i )*Horizon_SCAN;
                // 取第一个点云点上方的一个点云点 b
                upperInd = j + (i+1)*Horizon_SCAN;
                // 若一个点的强度值或者对应列上面的点等于-1？则将该点的标签写为-1
                //（该点的横纵坐标：thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;）
                if (fullCloud->points[lowerInd].intensity == -1 ||
                    fullCloud->points[upperInd].intensity == -1){
                    // no info to check, invalid points
                    groundMat.at<int8_t>(i,j) = -1;
                    continue;
                }
                // 计算 点a 和 点b之间的夹角ɑ   
                diffX = fullCloud->points[upperInd].x - fullCloud->points[lowerInd].x;
                diffY = fullCloud->points[upperInd].y - fullCloud->points[lowerInd].y;
                diffZ = fullCloud->points[upperInd].z - fullCloud->points[lowerInd].z;

                angle = atan2(diffZ, sqrt(diffX*diffX + diffY*diffY) ) * 180 / M_PI;
                // 如果夹角ɑ 小于设定阈值10，则将 点a 和 点b都标记为1（即地面点）
                if (abs(angle - sensorMountAngle) <= 10){
                    groundMat.at<int8_t>(i,j) = 1;
                    groundMat.at<int8_t>(i+1,j) = 1;
                }
            }
        }
        // extract ground cloud (groundMat == 1)
        // mark entry that doesn't need to label (ground and invalid point) for segmentation
        // 对于分割来说，地面和无效点不需要标记
        // note that ground remove is from 0~N_SCAN-1, need rangeMat for mark label matrix for the 16th scan

        // 遍历点云中所有的点，将地面点和无效点标记为-1
        for (size_t i = 0; i < N_SCAN; ++i){
            for (size_t j = 0; j < Horizon_SCAN; ++j){
                //在groundMat中将地面点和无效点 标记为-1
                if (groundMat.at<int8_t>(i,j) == 1 || rangeMat.at<float>(i,j) == FLT_MAX){
                    labelMat.at<int>(i,j) = -1;
                }
            }
        }
        //发布地面点云
        if (pubGroundCloud.getNumSubscribers() != 0){
            for (size_t i = 0; i <= groundScanInd; ++i){
                for (size_t j = 0; j < Horizon_SCAN; ++j){
                    if (groundMat.at<int8_t>(i,j) == 1)
                        groundCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                }
            }
        }
    }

    void cloudSegmentation(){
        // segmentation process
        //遍历点云中所有点
        for (size_t i = 0; i < N_SCAN; ++i)
            for (size_t j = 0; j < Horizon_SCAN; ++j)
                if (labelMat.at<int>(i,j) == 0)
                    labelComponents(i, j);
          
        int sizeOfSegCloud = 0;
        // extract segmented cloud for lidar odometry
        for (size_t i = 0; i < N_SCAN; ++i) {
//Q？        // 记录每一条scan中分割出的有效点起始index和终止index
            segMsg.startRingIndex[i] = sizeOfSegCloud-1 + 5;

            for (size_t j = 0; j < Horizon_SCAN; ++j) {
                if (labelMat.at<int>(i,j) > 0 || groundMat.at<int8_t>(i,j) == 1){
                    // outliers that will not be used for optimization (always continue)
                    if (labelMat.at<int>(i,j) == 999999){
                        //groundScanInd = 7
                        if (i > groundScanInd && j % 5 == 0){
                            outlierCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                            continue;
                        }else{
                            continue;
                        }
                    }
                    // majority of ground points are skipped
// Q?                    
                    if (groundMat.at<int8_t>(i,j) == 1){
                        if (j%5!=0 && j>5 && j<Horizon_SCAN-5)
                            continue;
                    }
                    // mark ground points so they will not be considered as edge features 
                    // 标记地面点，以便以后不会将其视为边缘特征
                    segMsg.segmentedCloudGroundFlag[sizeOfSegCloud] = (groundMat.at<int8_t>(i,j) == 1);
                    // mark the points' column index for marking occlusion later
                    // 标记点的列索引, 稍后标记遮挡
                    segMsg.segmentedCloudColInd[sizeOfSegCloud] = j;
                    // save range info
                    // 保存距离信息
                    segMsg.segmentedCloudRange[sizeOfSegCloud] = rangeMat.at<float>(i,j);
                    // save seg cloud
                    // 保存分割点云
                    segmentedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                    // size of seg cloud
                    // 更新分割点云数量
                    ++sizeOfSegCloud;
                }
            }

            segMsg.endRingIndex[i] = sizeOfSegCloud-1 - 5;
        }
        
        // extract segmented cloud for visualization
        // 可视化分割后的点云(同类点云分配相同的强度值)，不包括地面点云
        if (pubSegmentedCloudPure.getNumSubscribers() != 0){
            for (size_t i = 0; i < N_SCAN; ++i){
                for (size_t j = 0; j < Horizon_SCAN; ++j){
                    if (labelMat.at<int>(i,j) > 0 && labelMat.at<int>(i,j) != 999999){
                        // 给点云分配强度值: 同类点云, 强度值相同
                        segmentedCloudPure->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                        segmentedCloudPure->points.back().intensity = labelMat.at<int>(i,j);
                    }
                }
            }
        }
    }

    void labelComponents(int row, int col){
        // use std::queue std::vector std::deque will slow the program down greatly
        float d1, d2, alpha, angle;
        int fromIndX, fromIndY, thisIndX, thisIndY; 
        bool lineCountFlag[N_SCAN] = {false};

        queueIndX[0] = row;
        queueIndY[0] = col;
        int queueSize = 1;
        int queueStartInd = 0;
        int queueEndInd = 1;

        allPushedIndX[0] = row;
        allPushedIndY[0] = col;
        int allPushedIndSize = 1;
        
        while(queueSize > 0){
            // Pop point
            fromIndX = queueIndX[queueStartInd];
            fromIndY = queueIndY[queueStartInd];
            --queueSize;
            ++queueStartInd;
            // Mark popped point
            labelMat.at<int>(fromIndX, fromIndY) = labelCount;
            // Loop through all the neighboring grids of popped grid
            for (auto iter = neighborIterator.begin(); iter != neighborIterator.end(); ++iter){
                // new index
                thisIndX = fromIndX + (*iter).first;
                thisIndY = fromIndY + (*iter).second;
                // index should be within the boundary
                // 邻近的点应该在界限内（即x>=0 && x<16）
                if (thisIndX < 0 || thisIndX >= N_SCAN)
                    continue;
                // at range image margin (left or right side)
                // 当选取的点在距离图像的边缘时，通过该方法
                // 将距离图像的边缘连接起来
                if (thisIndY < 0)
                    thisIndY = Horizon_SCAN - 1;
                if (thisIndY >= Horizon_SCAN)
                    thisIndY = 0;
                // prevent infinite loop (caused by put already examined point back)
                // 防止死循环（把已经检查过的点放回去造成的）
                // 检查这个邻近点是否等于0，如果不等于说明已经贴完标签，贴完则跳过该点。
                if (labelMat.at<int>(thisIndX, thisIndY) != 0)
                    continue;
                
                // 该while循环到此，已经完成判断，获得该点的一个邻近点。
                // 此后比较两点距离，距离长的是d1，距离短的是d2            
                d1 = std::max(rangeMat.at<float>(fromIndX, fromIndY),  
                              rangeMat.at<float>(thisIndX, thisIndY));
                d2 = std::min(rangeMat.at<float>(fromIndX, fromIndY), 
                              rangeMat.at<float>(thisIndX, thisIndY));
                // 该邻近点和该点处于统一列
                if ((*iter).first == 0)
                    alpha = segmentAlphaX; // segmentAlphaX = 0.2 / 180 * PI
                // 该邻近点和该点处于统一行
                else
                    alpha = segmentAlphaY; // segmentAlphaY = 2.0 / 180 * PI
                // 计算相邻两点的角度
                angle = atan2(d2*sin(alpha), (d1 - d2*cos(alpha)));
                // 若该角度大于设定的阈值segmentTheta PI/3（60.0/180.0*M_PI）
                // 如果angle大于60°
                if (angle > segmentTheta){
                    
                    queueIndX[queueEndInd] = thisIndX;
                    queueIndY[queueEndInd] = thisIndY;
                    ++queueSize;
                    ++queueEndInd;
                    //给邻近点也贴上与中心点相同的标签
                    labelMat.at<int>(thisIndX, thisIndY) = labelCount;
                    lineCountFlag[thisIndX] = true;

                    allPushedIndX[allPushedIndSize] = thisIndX;
                    allPushedIndY[allPushedIndSize] = thisIndY;
                    ++allPushedIndSize;
                }
            }
        }

        // check if this segment is valid
        // 分割得到的cluster点集 >= 30
        bool feasibleSegment = false;
        if (allPushedIndSize >= 30)
            feasibleSegment = true;
        // segmentValidPointNum = 5
        // 分割得到的cluster点集,cluster中点的数量 >= 5 && 垂直方向上的线数 >= 3
        else if (allPushedIndSize >= segmentValidPointNum){
            int lineCount = 0;
            for (size_t i = 0; i < N_SCAN; ++i)
                if (lineCountFlag[i] == true)
                    ++lineCount;
            // segmentValidLineNum = 3
            if (lineCount >= segmentValidLineNum)
                feasibleSegment = true;            
        }
        // 同类点云具有相同的标号，噪声的标号为 999999
        // segment is valid, mark these points
        if (feasibleSegment == true){
            ++labelCount;
        }else{ // segment is invalid, mark these points
            for (size_t i = 0; i < allPushedIndSize; ++i){
                labelMat.at<int>(allPushedIndX[i], allPushedIndY[i]) = 999999;
            }
        }
    }

    
    void publishCloud(){
        // 1. Publish Seg Cloud Info
        // 发布点云，包括采样后的地面点和分割后的点云
        segMsg.header = cloudHeader;
        pubSegmentedCloudInfo.publish(segMsg);
        // 2. Publish clouds
        // 发布离群点
        sensor_msgs::PointCloud2 laserCloudTemp;

        pcl::toROSMsg(*outlierCloud, laserCloudTemp);
        laserCloudTemp.header.stamp = cloudHeader.stamp;
        laserCloudTemp.header.frame_id = "base_link";
        pubOutlierCloud.publish(laserCloudTemp);
        // segmented cloud with ground
        pcl::toROSMsg(*segmentedCloud, laserCloudTemp);
        laserCloudTemp.header.stamp = cloudHeader.stamp;
        laserCloudTemp.header.frame_id = "base_link";
        pubSegmentedCloud.publish(laserCloudTemp);
        // projected full cloud
        // 发布投影后的点云（坐标）
        if (pubFullCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*fullCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubFullCloud.publish(laserCloudTemp);
        }
        // original dense ground 
        // 发布地面点云
        if (pubGroundCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*groundCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubGroundCloud.publish(laserCloudTemp);
        }
        // segmented cloud without ground
        // 发布分割点云
        if (pubSegmentedCloudPure.getNumSubscribers() != 0){
            pcl::toROSMsg(*segmentedCloudPure, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubSegmentedCloudPure.publish(laserCloudTemp);
        }
        // projected full cloud info
        // 发布投影后的点云（距离）
        if (pubFullInfoCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*fullInfoCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubFullInfoCloud.publish(laserCloudTemp);
        }
    }
};




int main(int argc, char** argv){

    ros::init(argc, argv, "lego_loam");
    
    ImageProjection IP;

    ROS_INFO("\033[1;32m---->\033[0m Image Projection Started.");

    ros::spin();
    return 0;
}
