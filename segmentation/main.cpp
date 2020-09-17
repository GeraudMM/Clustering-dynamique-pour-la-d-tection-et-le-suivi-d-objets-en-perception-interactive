// Libraries
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <boost/make_shared.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/crop_box.h>
#include <pcl/point_types.h>
#include <pcl/features/fpfh.h>
#include <pcl/pcl_base.h>

#include <iostream>
#include <sstream>
#include <string>
#include <cmath>
#include <zmq.hpp>
#include <chrono>
#include <thread>
#include <math.h>

// Local libs includes
#include "libs/codelibrary/base/log.h"
#include "libs/codelibrary/geometry/point_cloud/supervoxel_segmentation.h"
#include "libs/papon/supervoxel/sequential_supervoxel_clustering.h"
#include "libs/supervoxel_tracker/supervoxel_tracker.h"
#include "libs/pairwise_segmentation/pairwise_segmentation.h"
#include "libs/image_processing/HistogramFactory.cpp"
#include "libs/image_processing/tools.cpp"


// Types
typedef pcl::tracking::ParticleXYZRPY StateT;
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointNormal PointXYZN;
typedef pcl::PointCloud<PointXYZN> PointCloudN;
typedef pcl::Normal Normal;
typedef pcl::PointCloud<Normal> NormalCloud;
typedef pcl::PointXYZL PointLT;
typedef pcl::PointCloud<PointLT> PointLCloudT;
typedef pcl::PointXYZRGBL PointRGBLT;
typedef pcl::PointCloud<PointRGBLT> PointRGBLCloudT;


using namespace std;
using byte = unsigned char;

/// Point with Normal.
struct PointWithNormal : cl::RPoint3D {
    PointWithNormal() {}

    cl::RVector3D normal;
};




int main() {
	float voxel_resolution = 0.008f;//Largeur d'un Voxel en mètres
	float seed_resolution = 0.08f;//Largeur d'un SuperVoxel en mètres
	float color_importance = 0.2f;
	float spatial_importance = 0.4f;
	float normal_importance = 1.0f;
	uint64_t min_number_in_radius_for_noise_reduction = 20;

	// Pointcloud, used to store the cloud from the rgbd camera
    PointCloudT::Ptr cloud (new PointCloudT);


	const std::string environment = "Environment3";	 //Environment1 is easy, 2 medium and 3 is hard
	const std::string table = "Table3";				             //Table1 is easy, 2 medium and 3 is hard
	const std::string nomDossier = "env3_table3";		 //Dossier dans lequel nous sauvegardons tous les fichiers
	//Attention, il faudra changer le dossier de sauvegarde si l'on ne veut pas ecraser les données des nuages de points capturés précedement
	//De plus il faut créer ce dossier manuellement dans le dossier output avant de lancer l'experience



	cl::Array<cl::RPoint3D> points(424*512);
	cl::Array<cl::RPoint3D> colors(424*512);
	cl::Array<int> pushables(424*512);

	LOG_ON(INFO);
    using namespace std::chrono_literals;

    // initialize the zmq context with a single IO thread
    zmq::context_t context{1};

    // construct a REP (reply) socket and bind to interface for communication with the Unity Part
    zmq::socket_t socket{context, zmq::socket_type::rep};
    socket.bind("tcp://*:5560");
    // construct a REQ (request) socket and bind to interface for communication with the Python Part
	zmq::socket_t requester(context, ZMQ_REQ);
    requester.connect("tcp://localhost:5562");

    // send the nbBatch request to python
	LOG(INFO) <<"Send Request nbBatch to Python...";
	const std::string pythonRequestnbBatch = "nbBatch";
    const std::string PyReqnbBatch{pythonRequestnbBatch};
	requester.send(zmq::buffer(PyReqnbBatch), zmq::send_flags::none);

	// Get the nbBatch reply from python
	zmq::message_t replynbBatch;
	requester.recv(replynbBatch, zmq::recv_flags::none);
	std::string repnbBatch = std::string(static_cast<char*>(replynbBatch.data()), replynbBatch.size());
	int nbBatch = 0;
	int accnbBatch = 0;
	for(int i=(int)repnbBatch.size(); i>0 ; i--){
		//A simplifier (ici on recoit un int et pas un char d'ou le -48 pour que 0 egal 0)
		nbBatch += (int)int(repnbBatch[i-1]-48)*pow(10,accnbBatch);
		accnbBatch+=1;
	}
	LOG(INFO) <<"nbBatch : "<< nbBatch;

    // send the nbScenetoConvert request to python
	LOG(INFO) <<"Send Request nbScenetoConvert to Python...";
	const std::string pythonRequestnbScenetoConvert = "nbScenetoConvert";
    const std::string PyReqnbScenetoConvert{pythonRequestnbScenetoConvert};
	requester.send(zmq::buffer(PyReqnbScenetoConvert), zmq::send_flags::none);

	// Get the nbScenetoConvert reply from python
	zmq::message_t replynbScenetoConvert;
	requester.recv(replynbScenetoConvert, zmq::recv_flags::none);
	std::string repnbScenetoConvert = std::string(static_cast<char*>(replynbScenetoConvert.data()), replynbScenetoConvert.size());
	int nbScenetoConvert = 0;
	int accnbScenetoConvert = 0;
	for(int i=(int)repnbScenetoConvert.size(); i>0 ; i--){
		//A simplifier (ici on recoit un int et pas un char d'ou le -48 pour que 0 egal 0)
		nbScenetoConvert += (int)int(repnbScenetoConvert[i-1]-48)*pow(10,accnbScenetoConvert);
		accnbScenetoConvert+=1;
	}
	LOG(INFO) <<"nbScenetoConvert : "<< nbScenetoConvert;

    // send the nbInteractionParScene request to python
	LOG(INFO) <<"Send Request nbInteractionParScene to Python...";
	const std::string pythonRequestnbInteractionParScene = "nbInteractionParScene";
    const std::string PyReqnbInteractionParScene{pythonRequestnbInteractionParScene};
	requester.send(zmq::buffer(PyReqnbInteractionParScene), zmq::send_flags::none);

	// Get the nbInteractionParScene reply from python
	zmq::message_t replynbInteractionParScene;
	requester.recv(replynbInteractionParScene, zmq::recv_flags::none);
	std::string repnbInteractionParScene = std::string(static_cast<char*>(replynbInteractionParScene.data()), replynbInteractionParScene.size());
	int nbInteractionParScene = 0;
	int accnbInteractionParScene = 0;
	for(int i=(int)repnbInteractionParScene.size(); i>0 ; i--){
		//A simplifier (ici on recoit un int et pas un char d'ou le -48 pour que 0 egal 0)
		nbInteractionParScene += (int)int(repnbInteractionParScene[i-1]-48)*pow(10,accnbInteractionParScene);
		accnbInteractionParScene+=1;
	}
	LOG(INFO) <<"nbInteractionParScene : "<< nbInteractionParScene;


	// Une fois qu'on a récupérer les données ci-dessus envoyé par Python, nous les envoyons à Unity aussi
	const std::string stringNbBatch = to_string(nbBatch);
	const std::string stringNbScenetoConvert = to_string(nbScenetoConvert);
	const std::string stringNbInteractionParScene = to_string(nbInteractionParScene);
	const std::string dataToSend = environment+","+table+","+stringNbBatch+","+stringNbScenetoConvert+","+stringNbInteractionParScene;
    const std::string data{dataToSend};

    //first exchange to send NbScene and choosen environment
    zmq::message_t request;
    // receive a request from client
    socket.recv(request, zmq::recv_flags::none);
    // send the reply to the Unity client
    socket.send(zmq::buffer(data), zmq::send_flags::none);

	//Début de la boucle de l'experience
	for(int nBatch = 1; nBatch<nbBatch+1; nBatch++){
		
		for (int nScene = 1; nScene<nbScenetoConvert+1; nScene++){

			// Create a supervoxel clustering instance
			pcl::SequentialSVClustering<PointT> super (voxel_resolution, seed_resolution);

			// Setting the importance of each parameter in feature space
			super.setColorImportance(color_importance);
			super.setSpatialImportance(spatial_importance);
			super.setNormalImportance(normal_importance);

			// Create supervoxel clusters
			pcl::SequentialSVClustering<PointT>::SequentialSVMapT supervoxel_clusters;
			supervoxel_clusters.clear ();

			for(int nIterScene = 1;nIterScene<nbInteractionParScene+2; nIterScene++){

				LOG(INFO) << "Creation de l'Experience Numero "<<nBatch<< " Scene Numero "<<nScene<<" Iteration Numero "<<nIterScene;
				LOG(INFO) << "";
				const std::string stringNScene = to_string(nScene);

				//Fichier comprenant les histogrammes de chaques supervoxels
				const std::string filenameFPFH = "../../outputs/"+nomDossier+"/fpfh/fpfh_scene"+stringNScene+"iter"+to_string(nIterScene)+".txt";
				//Fichier comprenant les voxels
				const std::string filenameVCCS = "../../outputs/"+nomDossier+"/vccs/vccs_scene"+stringNScene+"iter"+to_string(nIterScene)+".xyzrgbls";
				//Fichier comprenant les centroides des SVP sur une map lisible
				const std::string filenameCentroide = "../../outputs/"+nomDossier+"/centroides/centroides_scene"+stringNScene+"iter"+to_string(nIterScene)+".xyzrgbls";
				//Fichier Nuage 2D pour afficher plus simplement sur Python
				const std::string filenameNuage = "../../outputs/"+nomDossier+"/nuageDisplay/nuage_scene"+stringNScene+"iter"+to_string(nIterScene)+".xyzrgbls";

				LOG(INFO) << "debut Importation";
				zmq::message_t request;
				// receive a request from client
				socket.recv(request, zmq::recv_flags::none);
				LOG(INFO) << "Reading points from Unity...";
				std::string req = std::string(static_cast<char*>(request.data()), request.size());
				int n_points = (int)(req.size()/7);//Chaque points envoyé par Unity à 7 dimensions: RGBXYZL(L pour Label)
				LOG(INFO) << n_points << " points are imported";
				cloud.reset (new PointCloudT);
				
				std::ofstream nuage_file(filenameNuage);//Pour la sauvegarde du nuage de points dans un fichier
				nuage_file << std::setprecision(5);
				
				for(int i =0;i<(int)(req.size()/7);i++){
					int x = int((req[i*7+3]+256))%256;//On code les valeurs Unity sur 8bit chacune
					int y = int((req[i*7+4]+256))%256;
					int z = int((req[i*7+5]+256))%256;
					
					colors[i].x = int((req[i*7]+256))%256;          //R
					colors[i].y = int((req[i*7+1]+256))%256;     //G
					colors[i].z = int((req[i*7+2]+256))%256;     //B
					points[i].x = x;       										    //X
					points[i].y = y;        										//Y
					points[i].z = z;        										//Z
					pushables[i] = int(req[i*7+6]);        				//L (si le point en question est poussable, L = 1, sinon L = 0)

					pcl::PointXYZRGBA pt;
					pt.x = (float)(x)/128.f - 1.f;//range(-1;1)
					pt.y = (float)(y)/128.f - 1.f;//range(-1;1)
					pt.z = (float)(z)/128.f - 1.f;//range(-1;1)
					pt.r = int((req[i*7]+256))%256;//range(0;255)
					pt.g = int((req[i*7+1]+256))%256;//range(0;255)
					pt.b = int((req[i*7+2]+256))%256;//range(0;255)
					cloud->push_back(pt);

					nuage_file <<(int)(i%512)<<" ";                  //X
					nuage_file <<(int)(i/512)<<" ";                    //Y
					nuage_file <<(int)(points[i].z)<<" ";            //Z
					nuage_file <<(int)colors[i].x<<" ";              //R
					nuage_file <<(int)colors[i].y<<" ";              //G
					nuage_file <<(int)colors[i].z<<" ";              //B
					nuage_file <<"0"<<" ";                                 //L = 0 mais l'information n'est pas perdu, elle transite par un autre fichier
					nuage_file <<"0"<<"\n";                               //S = 0 mais l'information n'est pas perdu, elle transite par un autre fichier
				}
				nuage_file.close();




				
				// Filter the noise from the input pointcloud
				pcl::RadiusOutlierRemoval<PointT> rorfilter;
				rorfilter.setInputCloud (cloud);
				rorfilter.setRadiusSearch (2*voxel_resolution);
				rorfilter.setMinNeighborsInRadius(min_number_in_radius_for_noise_reduction);
				rorfilter.filter (*cloud);

				super.setInputCloud(cloud);
				LOG(INFO) <<"Extracting supervoxels!\n";
				super.extract (supervoxel_clusters);
				int n2_supervoxels = supervoxel_clusters.size ();
				LOG(INFO) << "Total SuperVoxel Persistant : "<<n2_supervoxels;
                                
				pcl::PointCloud<pcl::PointXYZ>::Ptr points_sv (new pcl::PointCloud<pcl::PointXYZ>);//il y a aussi des points XYZRGB si nécessaire
				pcl::PointCloud<pcl::Normal>::Ptr normals_sv (new pcl::PointCloud<pcl::Normal> ());
				// Create the FPFH estimation class, and pass the input dataset+normals to it
				pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
				
				int total_voxels = 0;
				cl::Array<cl::RPoint3D> points2_sv(n2_supervoxels);
				cl::Array<float> is_pushable_sv(n2_supervoxels);
				cl::Array<cl::RPoint3D> RGB_sv(n2_supervoxels);
				Eigen::VectorXd sample(15*n2_supervoxels);//Pour les Histogrammes CieLAB
				int accu = 0;
				std::ofstream vccs_file(filenameVCCS);
				vccs_file << std::setprecision(5);
				std::ofstream centroides_file(filenameCentroide);
				centroides_file << std::setprecision(5);
				
				for (const auto& sv: supervoxel_clusters)//On fait défiler les supervoxels dans une boucle
				{	
					int nb_voxels = sv.second->voxels_->points.size();
					total_voxels += nb_voxels;
					
					pcl::PointXYZ pt;
					pt.x = sv.second->centroid_.x;
					pt.y = sv.second->centroid_.y;
					pt.z = sv.second->centroid_.z;
					points_sv->push_back(pt);
					points2_sv[accu].x = sv.second->centroid_.x;
					points2_sv[accu].y = sv.second->centroid_.y;
					points2_sv[accu].z = sv.second->centroid_.z;
					
					pcl::Normal nrml;
					nrml.normal_x = sv.second->normal_.normal_x;
					nrml.normal_y = sv.second->normal_.normal_y;
					nrml.normal_z = sv.second->normal_.normal_z;
					normals_sv->push_back(nrml);	

					RGB_sv[accu].x = sv.second->centroid_.r;
					RGB_sv[accu].y = sv.second->centroid_.g;
					RGB_sv[accu].z = sv.second->centroid_.b;
					
					//Ici on retrouve les véritables labels de chauqe supervoxel ainsi que les numéro de supervoxel auquel appartient chaque point
					int is_pushable = 0;
					int i_iter = 0;					
					float min_dist  = 10000.0f;
					for(int pts_i=0;pts_i<424;pts_i+=3){//On peut etre plus précis en avancant 1 par 1 et non 3 par 3 mais cela est plus long
						for(int pts_j=0;pts_j<512;pts_j+=3){
							int pt_actuel = pts_i*512+pts_j;
							float dist = sqrt(pow(points[pt_actuel].x-(pt.x+1.f)*128,2)+pow(points[pt_actuel].y-(pt.y+1.f)*128,2)
								+pow(points[pt_actuel].z-(pt.z+1.f)*128,2));
							if(dist<min_dist){
								is_pushable = pushables[pt_actuel];
								min_dist = dist;
								i_iter = pt_actuel;
							}
						}
					}
					centroides_file <<(int)(i_iter%512)<<" ";             //X
					centroides_file <<(int)(i_iter/512)<<" ";               //Y
					centroides_file <<(int)(points[i_iter].z)<<" ";       //Z
					centroides_file <<(int)colors[i_iter].x<<" ";         //R
					centroides_file <<(int)colors[i_iter].y<<" ";         //G
					centroides_file <<(int)colors[i_iter].z<<" ";         //B
					centroides_file <<"0"<<" ";                                   //L
					centroides_file <<accu+1<<"\n";                          //S
					
						
						
					//Calcul des Histogrammes CieLAB15
					std::vector<Eigen::VectorXd> data;
					for(auto it = sv.second->voxels_->begin(); it != sv.second->voxels_->end(); ++it){
						float Lab[3];
						image_processing::tools::rgb2Lab(it->r,it->g,it->b,Lab[0],Lab[1],Lab[2]);
						Eigen::VectorXd vect(3);
						vect(0) = Lab[0]/100.0f;
						vect(1) = Lab[1]/128.0f;
						vect[2] = Lab[2]/128.0f;
						data.push_back(vect);
					}
					Eigen::MatrixXd bounds(2,3);
					bounds << 0,-1,-1,
							  1,1,1;
					image_processing::HistogramFactory hf(5,3,bounds);
					hf.compute(data);
					
					int k = 0 , l = 0;
					for(int hi = 0; hi < 15; hi++){
						sample(hi+15*accu) = hf.get_histogram()[k](l);
						l = (l+1)%5;
						if(l == 0)
							k++;						
					}
					

					is_pushable_sv[accu] = is_pushable;
					for(int vxl=0;vxl<nb_voxels;vxl++){
						vccs_file <<(int)((sv.second->voxels_->points[vxl].x+1.f)*(float)128.f)<<" ";        //X
						vccs_file <<(int)((sv.second->voxels_->points[vxl].y+1.f)*(float)128.f)<<" ";        //Y
						vccs_file <<(int)((sv.second->voxels_->points[vxl].z+1.f)*(float)128.f)<<" ";        //Z
						vccs_file <<(int)sv.second->voxels_->points[vxl].r<<" ";                             			//R
						vccs_file <<(int)sv.second->voxels_->points[vxl].g<<" ";                             			//G
						vccs_file <<(int)sv.second->voxels_->points[vxl].b<<" ";                            		    //B
						vccs_file <<"0"<<" ";                                                                									//L
						vccs_file <<accu+1<<"\n";                                                            								//S
					}
					accu +=1;
				}
				
				vccs_file.close();
				centroides_file.close();
				LOG(INFO) << "Total voxel :  "<<total_voxels;
				
				// Interactive segmentation
				std::vector <uint32_t> to_reset_parts = super.getToResetParts ();
				std::vector <uint32_t> moving_parts = super.getMovingParts ();

				fpfh.setInputCloud (points_sv);
				fpfh.setInputNormals (normals_sv);

				// Create an empty kdtree representation, and pass it to the FPFH estimation object.
				// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
				pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
				fpfh.setSearchMethod (tree);

				// Output datasets
				pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs (new pcl::PointCloud<pcl::FPFHSignature33> ());

				// Use all neighbors in a sphere of radius Xcm
				// IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
				fpfh.setRadiusSearch (seed_resolution*1.5);//20.0);//Should be identical to the segmentation resolution
				
				// Compute the features
				fpfh.compute (*fpfhs);

				std::string FPFHL = "";
				//write the histogram in a file
				LOG(INFO) << "Sauvegarde des fpfhs labelisees...";
				if(nIterScene!=1){
					FPFHL+= (char)moving_parts.size();//Nombre de SVP ayant été déplacés
					for(int moving_sv = 0; moving_sv<moving_parts.size(); moving_sv++){
						FPFHL+= (char)((int)moving_parts[moving_sv]/255);
						FPFHL+= (char)((int)moving_parts[moving_sv]%255);
					}
				}
				else{FPFHL+= (char)(-1);}//Si aucun objet n'es déplacé entre les 2 nuages de points on envoie le code -1
					

				std::ofstream out(filenameFPFH);
				out << std::setprecision(5);
				for(int i=1;i<49;i++){
					out <<"h"<<i<<",";
				}

				out<<"label\n";
				for(int i = 0; i< n2_supervoxels;i++){//On écrit les vecteurs discriminants des SVP dans un fichier
					for(int j=0;j<33;j++){//FPFH du SuperVoxel Persistant
						out << fpfhs->points[i].histogram[j] <<",";
						FPFHL += (char)((int)(fpfhs->points[i].histogram[j]));
					}
					for(int j=0;j<15;j++){//Histogramme du SuperVoxel Persistant
						out << sample(j+15*i) <<",";
						FPFHL += (char)((int)(sample(j+15*i)));
					}					
					out <<is_pushable_sv[i]<<"\n";//Label Réel du SuperVoxel Persistant
					FPFHL += (char)((int)(is_pushable_sv[i]));
				}
				out.close();



				// send the request to python
				LOG(INFO) <<"Send FPFHL to Python...";//On envoie aussices vecteurs discriminants à Python pour analyses
				requester.send(zmq::buffer(FPFHL), zmq::send_flags::none);

				//  Get the reply from python
				zmq::message_t reply;//On récupère le Numéro du SVP selectionné pour la Push Primitive
				requester.recv(reply, zmq::recv_flags::none);
				std::string rep = std::string(static_cast<char*>(reply.data()), reply.size());
				int supVoxNb = 0;
				int acc = 0;
				for(int i=(int)rep.size(); i>0 ; i--){
					//A simplifier (ici on recoit un int et pas un char d'ou le -48 pour que 0 egal 0)
					supVoxNb += (int)int(rep[i-1]-48)*pow(10,acc);
					acc+=1;
				}
				
				LOG(INFO) <<"message from Python sending to Unity : "<< supVoxNb<<"\n";
				std::string coordinates_to_hit;
				coordinates_to_hit = to_string((int)((points2_sv[supVoxNb].x+1.f)*(float)128.f));
				coordinates_to_hit +=",";
				coordinates_to_hit += to_string((int)((points2_sv[supVoxNb].y+1.f)*(float)128.f));
				coordinates_to_hit +=",";
				coordinates_to_hit += to_string((int)((points2_sv[supVoxNb].z+1.f)*(float)128.f));
				// send the reply to the client
				socket.send(zmq::buffer(coordinates_to_hit), zmq::send_flags::none);//On envoie les coordonnées pour la push primitive à Unity

			}
		}
	}
    return 0;
}
