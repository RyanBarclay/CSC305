/*
Name: Ryan Barclay
V#: V00842513
Code was running on Mac OS 12.1
Compiler: cmake/3.22.1/bin/cmake


Notes:
To toggle the first part and second part please look in the global variables. You can also toggle the bonus parts for bonus 2 and bonus 3 in there aswell.


*/

////////////////////////////////////////////////////////////////////////////////
// C++ include
#include <iostream>
#include <string>
#include <vector>
#include <limits>
#include <fstream>
#include <algorithm>
#include <numeric>

// Utilities for the Assignment
#include "utils.h"

// Image writing library
#define STB_IMAGE_WRITE_IMPLEMENTATION // Do not include this line twice in your project!
#include "stb_image_write.h"

// Shortcut to avoid Eigen:: everywhere, DO NOT USE IN .h
using namespace Eigen;

////////////////////////////////////////////////////////////////////////////////
// Class to store tree
////////////////////////////////////////////////////////////////////////////////
class AABBTree
{
public:
    class Node
    {
    public:
        AlignedBox3d bbox;
        int parent;   // Index of the parent node (-1 for root)
        int left;     // Index of the left child (-1 for a leaf)
        int right;    // Index of the right child (-1 for a leaf)
        int triangle; // Index of the node triangle (-1 for internal nodes)
    };

    std::vector<Node> nodes;
    int root;

    AABBTree() = default;                           // Default empty constructor
    AABBTree(const MatrixXd &V, const MatrixXi &F); // Build a BVH from an existing mesh
};

////////////////////////////////////////////////////////////////////////////////
// Scene setup, global variables
////////////////////////////////////////////////////////////////////////////////
const std::string data_dir = DATA_DIR;
const std::string filename("raytrace.png");
const std::string mesh_filename(data_dir + "dragon.off");

// Toggle for Method1
const bool Method1 = false;
const bool Bonus2 = false;
const bool Bonus3 = false;

// Bonus 2 stuff
double ep = .0001;

// Bonus 3 stuff
int max_bounce = 5;

// Camera settings
const double focal_length = 2;
const double field_of_view = 0.7854; // 45 degrees
const bool is_perspective = true;
const Vector3d camera_position(0, 0, 2);

// Triangle Mesh
MatrixXd vertices; // n x 3 matrix (n points)
MatrixXi facets;   // m x 3 matrix (m triangles)
AABBTree bvh;

// Material for the object, same material for all objects
const Vector4d obj_ambient_color(0.0, 0.5, 0.0, 0);
const Vector4d obj_diffuse_color(0.5, 0.5, 0.5, 0);
const Vector4d obj_specular_color(0.2, 0.2, 0.2, 0);
const double obj_specular_exponent = 256.0;
const Vector4d obj_reflection_color(0.7, 0.7, 0.7, 0);

// Precomputed (or otherwise) gradient vectors at each grid node
const int grid_size = 20;
std::vector<std::vector<Vector2d>> grid;

// Lights
std::vector<Vector3d> light_positions;
std::vector<Vector4d> light_colors;
// Ambient light
const Vector4d ambient_light(0.2, 0.2, 0.2, 0);

// expose function
bool find_nearest_object(const Vector3d &ray_origin, const Vector3d &ray_direction, Vector3d &p, Vector3d &N);

// Fills the different arrays
void setup_scene()
{
    // Loads file
    std::ifstream in(mesh_filename);
    std::string token;
    in >> token;
    int nv, nf, ne;
    in >> nv >> nf >> ne;
    vertices.resize(nv, 3);
    facets.resize(nf, 3);
    for (int i = 0; i < nv; ++i)
    {
        in >> vertices(i, 0) >> vertices(i, 1) >> vertices(i, 2);
    }
    for (int i = 0; i < nf; ++i)
    {
        int s;
        in >> s >> facets(i, 0) >> facets(i, 1) >> facets(i, 2);
        assert(s == 3);
    }

    // Toggle for Bonus 3
    if (Bonus3 == false)
    {
        max_bounce = 0;
    }
    // setup tree
    if (Method1 == false)
    {
        bvh = AABBTree(vertices, facets);
    }

    // Lights
    light_positions.emplace_back(8, 8, 0);
    light_colors.emplace_back(16, 16, 16, 0);

    light_positions.emplace_back(6, -8, 0);
    light_colors.emplace_back(16, 16, 16, 0);

    light_positions.emplace_back(4, 8, 0);
    light_colors.emplace_back(16, 16, 16, 0);

    light_positions.emplace_back(2, -8, 0);
    light_colors.emplace_back(16, 16, 16, 0);

    light_positions.emplace_back(0, 8, 0);
    light_colors.emplace_back(16, 16, 16, 0);

    light_positions.emplace_back(-2, -8, 0);
    light_colors.emplace_back(16, 16, 16, 0);

    light_positions.emplace_back(-4, 8, 0);
    light_colors.emplace_back(16, 16, 16, 0);
}

////////////////////////////////////////////////////////////////////////////////
// Custom Functions
////////////////////////////////////////////////////////////////////////////////

// For Triangle
Vector3d getPointTri(Vector3d u_vector, Vector3d v_vector, Vector3d d_vector, Vector3d a_vector, Vector3d e_vector, double &t)
{
    // *** Make matrix A ***

    Matrix3d A;
    A << -u_vector, -v_vector, d_vector;
    // std::cout << "Here is the matrix A:\n"
    //           << A << std::endl;

    // *** Make vector from a-e ***

    Vector3d ae_vector = a_vector - e_vector;
    // std::cout << "Here is the vector a:\n"
    //           << a_vector << std::endl;
    // std::cout << "Here is the vector e:\n"
    //           << e_vector << std::endl;
    // std::cout << "Here is the vector ae:\n"
    //           << ae_vector << std::endl;

    // *** Calc the values for solution ****
    Vector3d solution_vector = A.colPivHouseholderQr().solve(ae_vector);

    // For some reason the left and right version of the equation are giving different points

    // Vector3d rightside = a_vector + (solution_vector(0) * u_vector) + (solution_vector(1) * v_vector);

    // Vector3d leftside = e_vector + (solution_vector(2) * d_vector);

    // if (leftside != rightside)
    // {
    //     printf("NOT GOOD\n");
    //     std::cout << "Here is the left:\n"
    //               << leftside << std::endl;
    //     std::cout << "Here is the right:\n"
    //               << rightside << std::endl;
    // }
    t = solution_vector(2);

    return a_vector + (solution_vector(0) * u_vector) + (solution_vector(1) * v_vector);
}

bool rayTri(Vector3d u_vector, Vector3d v_vector, Vector3d d_vector, Vector3d a_vector, Vector3d e_vector)
{
    // *** Make matrix A ***

    Matrix3d A;
    A << -u_vector, -v_vector, d_vector;
    // std::cout << "Here is the matrix A:\n"
    //           << A << std::endl;

    // *** Make vector from a-e ***

    Vector3d ae_vector = a_vector - e_vector;
    // std::cout << "Here is the vector a:\n"
    //           << a_vector << std::endl;
    // std::cout << "Here is the vector e:\n"
    //           << e_vector << std::endl;
    // std::cout << "Here is the vector ae:\n"
    //           << ae_vector << std::endl;

    // *** Calc the values for solution ****
    Vector3d solution_vector = A.colPivHouseholderQr().solve(ae_vector);
    // std::cout << "Here is the vector solution:\n"
    //           << solution_vector << std::endl;

    // *** Validate answer ***

    // above computes t, β, and γ

    // Check t
    if (solution_vector(2) < 0)
    {
        // t should not be behind the origin from where the ray was cast
        return false;
    }

    // Note: u == β, v == γ

    // Check 0 < u < 1
    // (β <0) or (β >1−γ) should return false
    if (solution_vector(0) < 0 || solution_vector(0) > (1 - solution_vector(1)))
    {
        // u should be between 0 and one (exclusive)
        return false;
    }

    // Check 0 < v < 1
    // (γ =< 0) or (γ > 1) should return false
    if (solution_vector(1) < 0 || solution_vector(1) > 1)
    {
        return false;
    }

    // *** If all checks good, return true. ***

    // Note: the problem with this function is that there is no validation for an infinite t. We assume this case is handled by the calc for the solution vector.

    return true;
}

// For making bvh
/*
int Recursive(const MatrixXd &V, const MatrixXi &F, std::vector<int> &F_index)
{

    // TODO: Update to make sure that we only get applicable centroids
    // I think we are done, will have to confirm

    // Compute the centroids of all the triangles in the input mesh
    MatrixXd centroids(F_index.size(), V.cols());
    centroids.setZero();
    for (int i = 0; i < F_index.size(); ++i)
    {
        for (int k = 0; k < F.cols(); ++k)
        {
            centroids.row(i) += V.row(F(F_index[i], k));
        }
        centroids.row(i) /= F.cols();
    }
    // std::cout
    //     << "Here is the F:\n"
    //     << F << std::endl;

    // std::cout
    //     << "Here is the centroids:\n"
    //     << centroids << std::endl;
    // printf("\nHere is F_index\n");
    // for (int i = 0; i < F_index.size(); i++)
    //     std::cout
    //         << F_index.at(i) << ' ';

    // TODO

    // Top-down approach.
    // Split each set of primitives into 2 sets of roughly equal size,
    // based on sorting the centroids along one direction or another.

    // 1. Split the input set of triangles into two sets S1 and S2.
    // 2. Recursively build the subtree T1 corresponding to S1.
    // 3. Recursively build the subtree T2 corresponding to S2.
    // 4. Update the box of the current node by merging boxes of the root of T1 and T2.
    // 5. Return the new root node R.

    // 1.
    // 1.1. Determine what x, y, or z axis the largest spanned by the centroids

    Vector3d span = centroids.colwise().maxCoeff().transpose() - centroids.colwise().minCoeff().transpose();
    int axis; // set to the axis that the largest spanned
    // determine if x, y, or z is largest spanned, stored in axis as an index
    if ((span[0] >= span[1]) && (span[0] >= span[2]))
    {
        axis = 0;
    }
    else if ((span[1] >= span[0]) && (span[1] >= span[2]))
    {
        axis = 1;
    }
    else if ((span[2] >= span[0]) && (span[2] >= span[1]))
    {
        axis = 2;
    }

    // std::cout
    //     << "Here is the axis:\n"
    //     << axis << std::endl;

    // 1.2. Sort centroid and F accordingly

    // set up an index array to reference the fulcrum, so it is not changed.
    // std::vector<int> F_index;
    // for (int i = 0; i < F_index.size(); i++)
    // {
    //     F_index.push_back(i);
    // }

    // Sort F_index by centroid using axis with largest span.
    for (int i = 0; i < centroids.rows() - 1; i++)
    {
        for (int k = 0; k < centroids.rows() - i - 1; k++)
        {
            if (centroids.row(k)[axis] > centroids.row(k + 1)[axis])
            {
                // swap for centroids
                Vector3d temp = centroids.row(k).transpose();
                centroids.row(k) = centroids.row(k + 1);
                centroids.row(k + 1) = temp.transpose();

                // swap for Fulcrum by index
                int temp2 = F_index[k];
                F_index[k] = F_index[k + 1];
                F_index[k + 1] = temp2;
            }
        }
    }

    // 1.3 split into S1 and S2

    std::vector<int> S1, S2;

    for (int i = 0; i < F_index.size(); i++)
    {
        if (i <= F_index.size() / 2)
        {
            S1.push_back(F_index[i]);
            // L = [0:half]
        }
        else
        {
            S2.push_back(F_index[i]);
            // R = [half:F.index.size()-1]
        }
    }
    // printf("\nHere is F_index\n");
    // for (int i = 0; i < F_index.size(); i++)
    //     std::cout
    //         << F_index.at(i) << ' ';

    // printf("\nHere is S1\n");
    // for (int i = 0; i < S1.size(); i++)
    //     std::cout << S1.at(i) << ' ';

    // printf("\nHere is S2\n");
    // for (int i = 0; i < S2.size(); i++)
    //     std::cout << S2.at(i) << ' ';

    // 2.
    // 2.1. Set up S1 and recursively call the left side so the memory stored is let go after computation.

    return -1;
}
*/
bool is_light_visible(const Vector3d &ray_origin, const Vector3d &ray_direction, const Vector3d &light_position)
{
    // TODO: Determine if the light is visible here
    // Use find_nearest_object
    Vector3d p, N;

    const int nearest_object = find_nearest_object(light_position, -ray_direction, p, N);

    // Hit nothing, this should never happen
    // Will not be positive unless it hits somthing on a positive path of ray direction
    if (nearest_object < 0)
    {
        return true;
    }

    // If somthing was hit, we need to know if it is between the light or not.
    // Because we know the ray_origin is going to be intersected with, all we really care about is if there was somthing hit before the ray_origin
    // So if the thing that in intersected with is our object we are good, otherwise it is behind another.
    if (p.isApprox(ray_origin, ep * 2))
    {
        return true;
    }
    else
    {
        return false;
    }

    return true;
}
////////////////////////////////////////////////////////////////////////////////
// BVH Code
////////////////////////////////////////////////////////////////////////////////

AlignedBox3d
bbox_from_triangle(const Vector3d &a, const Vector3d &b, const Vector3d &c)
{
    AlignedBox3d box;
    box.extend(a);
    box.extend(b);
    box.extend(c);
    return box;
}

/*
AABBTree::AABBTree(const MatrixXd &V, const MatrixXi &F)
{
    // F is the list of triangles with pointers
    // V is the vertices, ie the actual coords of the points of the triangle

    // to start the recursive call for making a AABB tree we want an array that can store an index for each elementF
    std::vector<int> F_index;
    for (int i = 0; i < F.rows(); i++)
    {
        F_index.push_back(i);
    }

    Recursive(V, F, F_index);
}
*/

// Could not figure out how to get Top-Down construction working, I have commented my attempt. The code below is my attempt of doing bottom up. I know that I am not partitioning it correctly, I am using some pseudo code from a study group to implement it.
AABBTree::AABBTree(const MatrixXd &V, const MatrixXi &F)
{
    // Compute the centroids of all the triangles in the input mesh
    MatrixXd centroids(F.rows(), V.cols());
    centroids.setZero();
    for (int i = 0; i < F.rows(); ++i)
    {
        for (int k = 0; k < F.cols(); ++k)
        {
            centroids.row(i) += V.row(F(i, k));
        }
        centroids.row(i) /= F.cols();
    }

    printf("Starting Init for AABB tree\n");

    // start of implementation

    std::vector<int> node_holder; // array of index's of the nodes to keep track of them

    // For each triangle (F.rows()) we want to init the node object for it and slap it in node_holder
    for (int i = 0; i < F.rows(); i++)
    {
        Node node_token;

        // init a,b,c
        Vector3d a, b, c;

        // Get points from triangle
        Vector3i verticesIndex;
        verticesIndex << facets.row(i).transpose();

        a << vertices.row(verticesIndex(0)).transpose();
        b << vertices.row(verticesIndex(1)).transpose();
        c << vertices.row(verticesIndex(2)).transpose();

        // Init bounding box
        node_token.bbox = bbox_from_triangle(a, b, c);

        // Init params of node
        node_token.triangle = i;
        node_token.left = -1; // default values
        node_token.right = -1;
        node_token.parent = -1;
        node_holder.push_back(i);

        // Add the node to the nodes in bvh object
        nodes.push_back(node_token);
    }

    while (node_holder.size() != 1)
    {
        for (int i = 0; i < node_holder.size() - 1; i++) // for every node in node_holder (triangle)
        {
            // For each triangle we want to find the pairing with the min distance between them
            // NOTE: I know that this is not the way to properly do it, it was a method that worked with peers so I have implemented it as such.
            // The actual attempt of this being done is commented out above in the attempt of Top-Down
            double cur_min_node_dist = std::numeric_limits<double>::max();
            int closesest_triangle = -1;

            for (int j = i + 1; j < node_holder.size(); j++) // for every other node left to the right of it
            {
                double dist = nodes[node_holder[i]].bbox.squaredExteriorDistance(nodes[node_holder[j]].bbox); // distance to the box
                // Praise god for squaredExteriorDistance and the legend that shouted this out in discord

                if (dist < cur_min_node_dist)
                {
                    cur_min_node_dist = dist; // set new min distance
                    closesest_triangle = j;   // Update with proper index

                    // Sanity check for closest distance, in case I am doing somthing wrong
                    if (cur_min_node_dist <= 0)
                    {
                        break;
                    }
                }
            }

            // creates a new bounding box that contains the 2 boxes at i and j
            Node doubled_up;

            doubled_up.bbox.extend(nodes[node_holder[i]].bbox);                  // update the bounding box for the first node
            doubled_up.bbox.extend(nodes[node_holder[closesest_triangle]].bbox); // update the bounding box for the second node

            // Init params of node
            doubled_up.left = node_holder[i];                   // set as one of the boxes
            doubled_up.right = node_holder[closesest_triangle]; // set as the other closest one
            doubled_up.triangle = -1;                           // default values
            doubled_up.parent = -1;

            // Add the new node to the bvh structure
            nodes.push_back(doubled_up);                                      // add bounding box to nodes
            nodes[node_holder[i]].parent = nodes.size() - 1;                  // make the i node parent the new root
            nodes[node_holder[closesest_triangle]].parent = nodes.size() - 1; // do the same for the j node

            node_holder[i] = nodes.size() - 1;                           // make i node in array the root node
            node_holder.erase(node_holder.begin() + closesest_triangle); // remove the j node from the array
        }
    }

    nodes[nodes.size() - 1].parent = -1; // set root node as -1 (this means root)
    root = nodes.size() - 1;             // set global root

    printf("AABB Tree Has been constructed\n");
}
////////////////////////////////////////////////////////////////////////////////
// Intersection code
////////////////////////////////////////////////////////////////////////////////

double ray_triangle_intersection(const Vector3d &ray_origin, const Vector3d &ray_direction, const Vector3d &a, const Vector3d &b, const Vector3d &c, Vector3d &p, Vector3d &N)
{
    // This returns the intersection between the ray and the triange at index index.
    // return t or -1 if no intersection

    // TODO
    // Compute whether the ray intersects the given triangle.
    // If you have done the parallelogram case, this should be very similar to it.

    // init local variables for calculation

    const Vector3d pgram_u = b - a;
    const Vector3d pgram_v = c - a;

    double t = -1;

    if (!rayTri(pgram_u, pgram_v, ray_direction, a, ray_origin))
    {
        return -1;
    }

    // TODO set the correct intersection point, update p and N to the correct values

    p = getPointTri(pgram_u, pgram_v, ray_direction, a, ray_origin, t);
    N = pgram_v.cross(pgram_u).normalized();

    return t;
}

bool ray_box_intersection(const Vector3d &ray_origin, const Vector3d &ray_direction, const AlignedBox3d &box)
{
    // TODO
    // Compute whether the ray intersects the given box.
    // we are not testing with the real surface here anyway.

    // Pseudocode from textbook.
    // Note: Only for a 2D example, as a 3D box will have 6 faces.
    // txMin = (xMin - xE)/ xD
    // txMax = (xMax - yE)/ yD
    // tyMin = (yMin - yE)/ yD
    // tyMax = (yMax - yE)/ yD
    // if (txMin > tyMax or tyMin > txMax):
    //      return false;
    // else:
    //      return true;

    float tMin = 0;                                 // we init to 0
    float tMax = std::numeric_limits<float>::max(); // we init to the max value

    for (int i = 0; i < 3; i++)
    {
        //  ray_direction = (xD, yD, zD)
        //  ray_origin = (xE, yE, zE)
        //  box.min() = (xMin, yMin, zMin)
        //  box.max() = (xMax, yMax, zMax)

        //  Now that we have identified the variables, now we can run a variation of the pseudocode on each axis, to account for the 3D structure of the box.

        float tiMin = (box.min()[i] - ray_origin[i]) / ray_direction[i];
        float tiMax = (box.max()[i] - ray_origin[i]) / ray_direction[i];

        // if ray_direction in a specifc axis is - we need to flip
        if (ray_direction[i] < 0.0)
        {
            std::swap(tiMin, tiMax);
        }

        // Determine correct max and min values for t
        tMin = std::max(tMin, tiMin);
        tMax = std::min(tMax, tiMax);

        if (tMin > tMax)
        {
            return false;
        }
    }
    return true;
}

// Finds the closest intersecting object returns its index
// In case of intersection it writes into p and N (intersection point and normals)
bool find_nearest_object(const Vector3d &ray_origin, const Vector3d &ray_direction, Vector3d &p, Vector3d &N)
{
    Vector3d tmp_p, tmp_N;

    // TODO
    // Method (1): Traverse every triangle and return the closest hit.

    // Set to false for toggling between method 1 and two. Look in global vars
    if (Method1 == true)
    {

        int closest_index = -1;
        double closest_t = std::numeric_limits<double>::max(); // closest t is "+ infinity"

        for (int i = 0; i < facets.rows(); ++i)
        {

            // init a,b,c
            Vector3d a, b, c;

            // Get points from triangle
            Vector3i verticesIndex;
            verticesIndex << facets.row(i).transpose();

            a << vertices.row(verticesIndex(0)).transpose();
            b << vertices.row(verticesIndex(1)).transpose();
            c << vertices.row(verticesIndex(2)).transpose();

            // printf("Index: i = %i\n", i);
            // std::cout
            //     << "Here is the verticesIndex:\n"
            //     << verticesIndex << std::endl;

            // std::cout << "Here is the a:\n"
            //           << a << std::endl;
            // std::cout << "Here is the b:\n"
            //           << b << std::endl;
            // std::cout << "Here is the c:\n"
            //           << c << std::endl;
            // printf("\n");

            // returns t and writes on tmp_p and tmp_N
            const double t = ray_triangle_intersection(ray_origin, ray_direction, a, b, c, tmp_p, tmp_N);
            // We have intersection
            if (t >= 0)
            {
                // The point is before our current closest t
                if (t < closest_t)
                {
                    closest_index = i;
                    closest_t = t;
                    p = tmp_p;
                    N = tmp_N;
                }
            }
        }

        if (closest_index != -1)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    else
    {

        // Method (2): Traverse the BVH tree and test the intersection with a
        // triangles at the leaf nodes that intersects the input ray.

        // Note: Based off an online tutorial that was sent to me from a study group on stack overflow. Sorry if it is not done 100% precisely

        int closest_index = -1;
        double closest_t = std::numeric_limits<double>::max(); // closest t is "+ infinity"

        // START OF IMPLEMENTATION

        std::vector<int> nodes_to_check; // this is a dynamic array for all the indexes of the nodes we care about
        int root_token = bvh.root;
        nodes_to_check.push_back(root_token); // To start we add the root so we can get going

        // Now for each of the nodes in this array we can check to see if it's a hit or a miss as we traverse the BVH
        while (nodes_to_check.size() > 0) // while still nodes to check
        {
            // Now we will check to see if the given list / vector is a hit or not.
            // Note: we will be mutating nodes_to_check as we go.
            for (int i = 0; i < nodes_to_check.size(); i++)
            {
                const AABBTree::Node &node_cur = bvh.nodes[nodes_to_check[i]]; // This is just used as a token for the node

                if (ray_box_intersection(ray_origin, ray_direction, node_cur.bbox))
                {
                    if (node_cur.triangle != -1) // will go here if not a internal, ie has to be a leaf
                    {
                        int triangle_index = node_cur.triangle;

                        // Using same code from method 1
                        Vector3d a, b, c;

                        // Get points from triangle
                        Vector3i verticesIndex;
                        verticesIndex << facets.row(triangle_index).transpose(); // get the triangle we care about

                        // init for point: a, b, c.
                        a << vertices.row(verticesIndex(0)).transpose();
                        b << vertices.row(verticesIndex(1)).transpose();
                        c << vertices.row(verticesIndex(2)).transpose();

                        // returns t and writes on tmp_p and tmp_N
                        const double t = ray_triangle_intersection(ray_origin, ray_direction, a, b, c, tmp_p, tmp_N);

                        if (t >= 0)
                        {
                            // The point is before our current closest t
                            if (t < closest_t)
                            {
                                closest_index = i;
                                closest_t = t;
                                p = tmp_p;
                                N = tmp_N;
                            }
                        }
                    }
                    else
                    {
                        // if the node_cur is not a leaf add on the left and right tree to the array
                        nodes_to_check.push_back(node_cur.left);
                        nodes_to_check.push_back(node_cur.right);
                    }
                }
                // If we do not intersect with the bounding box we want to erase the node from the list
                // Additionally we want to remove the node we have processed
                nodes_to_check.erase(nodes_to_check.begin() + i);
            }
        }

        if (closest_index != -1)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Raytracer code
////////////////////////////////////////////////////////////////////////////////

Vector4d shoot_ray(const Vector3d &ray_origin, const Vector3d &ray_direction, int max_bounce)
{
    // Intersection point and normal, these are output of find_nearest_object
    Vector3d p, N;

    const bool nearest_object = find_nearest_object(ray_origin, ray_direction, p, N);

    if (!nearest_object)
    {
        // Return a transparent color
        return Vector4d(0, 0, 0, 0);
    }

    // Ambient light contribution
    const Vector4d ambient_color = obj_ambient_color.array() * ambient_light.array();

    // Punctual lights contribution (direct lighting)
    Vector4d lights_color(0, 0, 0, 0);
    for (int i = 0; i < light_positions.size(); ++i)
    {
        const Vector3d &light_position = light_positions[i];
        const Vector4d &light_color = light_colors[i];

        const Vector3d Li = (light_position - p).normalized();

        // TODO: Shoot a shadow ray to determine if the light should affect the intersection point and call is_light_visible
        if (Bonus2 == true)
        {
            if (!is_light_visible(p, Li, light_position))
            {
                continue;
            }
        }

        Vector4d diff_color = obj_diffuse_color;

        // TODO: Add shading parameters

        // Diffuse contribution
        const Vector4d diffuse = diff_color * std::max(Li.dot(N), 0.0);

        // Specular contribution
        const Vector3d Hi = (Li - ray_direction).normalized();
        const Vector4d specular = obj_specular_color * std::pow(std::max(N.dot(Hi), 0.0), obj_specular_exponent);
        // Vector3d specular(0, 0, 0);

        // Attenuate lights according to the squared distance to the lights
        const Vector3d D = light_position - p;
        lights_color += (diffuse + specular).cwiseProduct(light_color) / D.squaredNorm();
    }

    // Bonus 3 stuff
    Vector4d refl_color = obj_reflection_color;
    refl_color = Vector4d(0.5, 0.5, 0.5, 0);
    // TODO: Compute the color of the reflected ray and add its contribution to the current point color.
    // use refl_color
    Vector4d reflection_color(0, 0, 0, 0);

    // We know that we will hit an object with the trajectory given at this point

    if (max_bounce != 0)
    {
        const Vector3d v = (ray_origin - p).normalized();
        const Vector3d r = ((2 * N * (N.dot(v))) - v).normalized();
        const Vector3d IMFUCKINGSTUPID = p + ep * r;

        reflection_color = (refl_color.cwiseProduct(shoot_ray(IMFUCKINGSTUPID, r, max_bounce - 1)));
    }

    // Rendering equation
    Vector4d C = ambient_color + lights_color + reflection_color;

    // Set alpha to 1
    C(3) = 1;

    return C;
}

////////////////////////////////////////////////////////////////////////////////

void raytrace_scene()
{
    std::cout << "Simple ray tracer." << std::endl;

    int w = 640;
    int h = 480;
    MatrixXd R = MatrixXd::Zero(w, h);
    MatrixXd G = MatrixXd::Zero(w, h);
    MatrixXd B = MatrixXd::Zero(w, h);
    MatrixXd A = MatrixXd::Zero(w, h); // Store the alpha mask

    // The camera always points in the direction -z
    // The sensor grid is at a distance 'focal_length' from the camera center,
    // and covers an viewing angle given by 'field_of_view'.
    double aspect_ratio = double(w) / double(h);

    // double image_y = 1; // TODO: compute the correct pixels size
    double image_y = tanf(field_of_view / 2) * focal_length;
    // double image_x = 1; // TODO: compute the correct pixels size
    double image_x = tanf(field_of_view / 2) * focal_length * aspect_ratio;

    // The pixel grid through which we shoot rays is at a distance 'focal_length'
    const Vector3d image_origin(-image_x, image_y, camera_position[2] - focal_length);
    const Vector3d x_displacement(2.0 / w * image_x, 0, 0);
    const Vector3d y_displacement(0, -2.0 / h * image_y, 0);

    for (unsigned i = 0; i < w; ++i)
    {
        for (unsigned j = 0; j < h; ++j)
        {
            const Vector3d pixel_center = image_origin + (i + 0.5) * x_displacement + (j + 0.5) * y_displacement;

            // Prepare the ray
            Vector3d ray_origin;
            Vector3d ray_direction;

            if (is_perspective)
            {
                // Perspective camera
                ray_origin = camera_position;
                ray_direction = (pixel_center - camera_position).normalized();
            }
            else
            {
                // Orthographic camera
                ray_origin = pixel_center;
                ray_direction = Vector3d(0, 0, -1);
            }

            const Vector4d C = shoot_ray(ray_origin, ray_direction, max_bounce);
            R(i, j) = C(0);
            G(i, j) = C(1);
            B(i, j) = C(2);
            A(i, j) = C(3);
        }
    }

    // Save to png
    write_matrix_to_png(R, G, B, A, filename);
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
    setup_scene();

    raytrace_scene();
    return 0;
}
