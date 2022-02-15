// Assignment 2
// Name: Ryan Barclay
// V#: V00842513
// Code was running on Mac OS V: 12.1
// Using compiler cmake/3.22.1/bin/cmake
// Description:

// Ex.1: Basic Ray Tracing - 7
/*
Compare the difference in the result for a sphere for a parallelogram

When I swapped between perspective to orthographic I noticed that the sphere got larger. As well the light / shaders were also behaving slightly differently, I assume this is perspective lighting. When swapping from perspective to orthographic the size increase is the same as the sphere. As well, the screen seems to almost cut off some of the parallelogram, I assume this is because of the orintation of the rays with the object when hitting at different angles.

*/

// Ex.2: Shading - 3

/*
Experiment with different parameters and observe their effect of the ray-traced shapes.

If you tweak where the object is it seems to affect size, and all aspects of shading except for ambient. If you mess with the exponent for the specular shading equation it seems that it amplifies the amount of light being reflected to the eye. What I find interesting is that with this code there is no way to "view" the light. So if we look at the formula for specular shading, we see that ks is between 0-1. Additionally the dot product could return a value more than one, so that means that the perspective light could actually be brighter than the actual light its self.

*/

// The rest of the steps/ requirements are done in the code below. Additionally, the parameters for the objects for all fuctions called give the provided image provided by the readme.

// C++ include
#include <iostream>
#include <string>
#include <vector>

// Utilities for the Assignment
#include "utils.h"

// Image writing library
#define STB_IMAGE_WRITE_IMPLEMENTATION // Do not include this line twice in your project!
#include "stb_image_write.h"

// Shortcut to avoid Eigen:: everywhere, DO NOT USE IN .h
using namespace Eigen;

// Custom Functions

// For Sphere
Vector3d getPointSphere(Vector3d c, Vector3d e, Vector3d d, double R)
{
    // getPointSphere will return a point in 3d space where a ray intersects with a sphere as long as there exists one valid t that is positive.
    // p(t) = e + t*d
    // f(p) = (p-c)*(p-c) - R^2

    // When we get here we know that the ray intersects with the sphere in a good way (t > 0) for at least one t.

    double A = d.dot(d);
    double B = (2 * d).dot(e - c);
    double C = (e - c).dot(e - c) - (R * R);

    double discriminant = (B * B) - (4 * A * C);

    double t;

    if (discriminant == 0)
    {
        // One intersection point
        t = (-B + sqrt(discriminant)) / (2 * A);
        // Because t is positive here we know that this t is good to go.
    }
    else if (discriminant > 0)
    {
        // More than one intersection point, in this case two
        t = (-B + sqrt(discriminant)) / (2 * A);
        double t2 = (-B - sqrt(discriminant)) / (2 * A);
        // Because in the check we know both or one of the t's are positive.
        if (t < 0)
        {
            t = t2;
            // if t is the negative one we know t2 is the positive
        }
        else if (t > 0 && t2 > 0)
        {
            // if both t's are valid for intersection then we gotta find the smaller one
            t = fmin(t, t2);
        }
    }
    // At this point we know the correct t for the intersect.

    return (e + (t * d));
}

bool raysphere(Vector3d c, Vector3d e, Vector3d d, double R)
{
    double A = d.dot(d);
    double B = (2 * d).dot(e - c);
    double C = (e - c).dot(e - c) - (R * R);

    double discriminant = (B * B) - (4 * A * C);

    if (discriminant < 0)
    {
        return false;
    }
    else if (discriminant == 0)
    {
        // One intersection point
        double t = (-B + sqrt(discriminant)) / (2 * A);

        if (t < 0)
        {
            return false;
        }
    }
    else if (discriminant > 0)
    {
        // More than one intersection point, in this case two
        double t = (-B + sqrt(discriminant)) / (2 * A);
        double t2 = (-B - sqrt(discriminant)) / (2 * A);

        if (t < 0 && t2 < 0)
        {
            return false;
        }
    }
    return true;
}

// For Parallelogram
Vector3d getPoint(Vector3d u_vector, Vector3d v_vector, Vector3d d_vector, Vector3d a_vector, Vector3d e_vector)
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

    return a_vector + (solution_vector(0) * u_vector) + (solution_vector(1) * v_vector);
}

bool raytri(Vector3d u_vector, Vector3d v_vector, Vector3d d_vector, Vector3d a_vector, Vector3d e_vector)
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

    // Check t
    if (solution_vector(2) < 0)
    {
        return false;
    }

    // Check u
    if (solution_vector(0) < 0 || solution_vector(0) > 1)
    {
        return false;
    }

    // Check 0 <= v <= 1
    if (solution_vector(1) < 0 || solution_vector(1) > 1)
    {
        return false;
    }

    // *** If all checks good, return true. ***
    return true;
}

void raytrace_sphere()
{
    std::cout << "Simple ray tracer, one sphere with orthographic projection" << std::endl;

    const std::string filename("sphere_orthographic.png");
    MatrixXd C = MatrixXd::Zero(800, 800); // Store the color
    MatrixXd A = MatrixXd::Zero(800, 800); // Store the alpha mask

    const Vector3d camera_origin(0, 0, 3);
    const Vector3d camera_view_direction(0, 0, -1);

    // The camera is orthographic, pointing in the direction -z and covering the
    // unit square (-1,1) in x and y
    const Vector3d image_origin(-1, 1, 1);
    const Vector3d x_displacement(2.0 / C.cols(), 0, 0);
    const Vector3d y_displacement(0, -2.0 / C.rows(), 0);

    // Single light source
    const Vector3d light_position(-1, 1, 1);

    for (unsigned i = 0; i < C.cols(); ++i)
    {
        for (unsigned j = 0; j < C.rows(); ++j)
        {
            const Vector3d pixel_center = image_origin + double(i) * x_displacement + double(j) * y_displacement;

            // Prepare the ray
            const Vector3d ray_origin = pixel_center;
            const Vector3d ray_direction = camera_view_direction;

            // Intersect with the sphere
            // NOTE: this is a special case of a sphere centered in the origin and for orthographic rays aligned with the z axis
            Vector2d ray_on_xy(ray_origin(0), ray_origin(1));
            const double sphere_radius = 0.9;

            if (ray_on_xy.norm() < sphere_radius)
            {
                // The ray hit the sphere, compute the exact intersection point
                Vector3d ray_intersection(
                    ray_on_xy(0), ray_on_xy(1),
                    sqrt(sphere_radius * sphere_radius - ray_on_xy.squaredNorm()));

                // Compute normal at the intersection point
                Vector3d ray_normal = ray_intersection.normalized();

                // Simple diffuse model
                C(i, j) = (light_position - ray_intersection).normalized().transpose() * ray_normal;

                // Clamp to zero
                C(i, j) = std::max(C(i, j), 0.);

                // Disable the alpha mask for this pixel
                A(i, j) = 1;
            }
        }
    }

    // Save to png
    write_matrix_to_png(C, C, C, A, filename);
}

void raytrace_parallelogram()
{
    std::cout << "Simple ray tracer, one parallelogram with orthographic projection" << std::endl;

    const std::string filename("plane_orthographic.png");
    MatrixXd C = MatrixXd::Zero(800, 800); // Store the color
    MatrixXd A = MatrixXd::Zero(800, 800); // Store the alpha mask

    const Vector3d camera_origin(0, 0, 3);
    const Vector3d camera_view_direction(0, 0, -1);

    // The camera is orthographic, pointing in the direction -z and covering the unit square (-1,1) in x and y
    const Vector3d image_origin(-1, 1, 1);
    const Vector3d x_displacement(2.0 / C.cols(), 0, 0);
    const Vector3d y_displacement(0, -2.0 / C.rows(), 0);

    // TODO: Parameters of the parallelogram (position of the lower-left corner + two sides)
    const Vector3d pgram_origin(-0.5, -0.5, 0); // a
    const Vector3d pgram_u(0, 0.7, -10);        // b - a
    const Vector3d pgram_v(1, 0.4, 0);          // c - a

    // Single light source
    const Vector3d light_position(-1, 1, 1);

    for (unsigned i = 0; i < C.cols(); ++i)
    {
        for (unsigned j = 0; j < C.rows(); ++j)
        {
            const Vector3d pixel_center = image_origin + double(i) * x_displacement + double(j) * y_displacement;

            // Prepare the ray
            const Vector3d ray_origin = pixel_center;             // e
            const Vector3d ray_direction = camera_view_direction; // d

            // TODO: Check if the ray intersects with the parallelogram

            // f(u,v) = a + (b-a)u + (c-a)v  **Will need to confirm if need to make unit vector length or not**
            // ** this is a (x,y,z) point that is inside of the parallelogram **
            // b = pu + a
            // c = pv + a
            // a = origin

            // p(t) = e + (t*d) ** this is a (x,y,z) point **
            // e = origin of ray
            // d = ray direction

            // Intersects if p(t) = f(u,v); {t > 0, u >= 0 , v >= 0, u+v <= 1}

            //

            if (raytri(pgram_u, pgram_v, ray_direction, pgram_origin, ray_origin))
            {
                // TODO: The ray hit the parallelogram, compute the exact intersection
                // point

                Vector3d ray_intersection = getPoint(pgram_u, pgram_v, ray_direction, pgram_origin, ray_origin);

                // TODO: Compute normal at the intersection point
                // Vector3d ray_normal = ray_intersection.normalized();
                Vector3d ray_normal = (pgram_v.cross(pgram_u)).normalized();

                // Simple diffuse model
                C(i, j) = (light_position - ray_intersection).normalized().transpose() * ray_normal;

                // Clamp to zero
                C(i, j) = std::max(C(i, j), 0.);

                // Disable the alpha mask for this pixel
                A(i, j) = 1;
            }
        }
    }

    // Save to png
    write_matrix_to_png(C, C, C, A, filename);
}

void raytrace_perspective()
{
    std::cout << "Simple ray tracer, one parallelogram with perspective projection" << std::endl;

    const std::string filename("plane_perspective.png");
    MatrixXd C = MatrixXd::Zero(800, 800); // Store the color
    MatrixXd A = MatrixXd::Zero(800, 800); // Store the alpha mask

    const Vector3d camera_origin(0, 0, 3);
    const Vector3d camera_view_direction(0, 0, -1);

    // The camera is perspective, pointing in the direction -z and covering the unit square (-1,1) in x and y
    const Vector3d image_origin(-1, 1, 1);
    const Vector3d x_displacement(2.0 / C.cols(), 0, 0);
    const Vector3d y_displacement(0, -2.0 / C.rows(), 0);

    // TODO: Parameters of the parallelogram (position of the lower-left corner + two sides)
    const Vector3d pgram_origin(-0.5, -0.5, 0);
    const Vector3d pgram_u(0, 0.7, -10);
    const Vector3d pgram_v(1, 0.4, 0);

    // Single light source
    const Vector3d light_position(-1, 1, 1);

    for (unsigned i = 0; i < C.cols(); ++i)
    {
        for (unsigned j = 0; j < C.rows(); ++j)
        {
            const Vector3d pixel_center = image_origin + double(i) * x_displacement + double(j) * y_displacement;

            // TODO: Prepare the ray (origin point and direction)
            // The ray will go from the camera origin to the pixel center and through

            // ray is starting from the camera
            const Vector3d ray_origin = camera_origin;
            // From camera origin to pixel
            const Vector3d ray_direction = pixel_center - camera_origin;

            // TODO: Check if the ray intersects with the parallelogram
            if (raytri(pgram_u, pgram_v, ray_direction, pgram_origin, ray_origin))
            {
                // TODO: The ray hit the parallelogram, compute the exact intersection
                // point

                Vector3d ray_intersection = getPoint(pgram_u, pgram_v, ray_direction, pgram_origin, ray_origin);

                // TODO: Compute normal at the intersection point
                // Vector3d ray_normal = ray_intersection.normalized();
                Vector3d ray_normal = (pgram_v.cross(pgram_u)).normalized();

                // Simple diffuse model
                C(i, j) = (light_position - ray_intersection).normalized().transpose() * ray_normal;

                // Clamp to zero
                C(i, j) = std::max(C(i, j), 0.);

                // Disable the alpha mask for this pixel
                A(i, j) = 1;
            }
        }
    }

    // Save to png
    write_matrix_to_png(C, C, C, A, filename);
}

void raytrace_shading()
{

    std::cout << "Simple ray tracer, one sphere with different shading" << std::endl;

    const std::string filename("shading.png");
    MatrixXd C = MatrixXd::Zero(800, 800); // Store the color
    MatrixXd R = MatrixXd::Zero(800, 800); // Store the color
    MatrixXd G = MatrixXd::Zero(800, 800); // Store the color
    MatrixXd B = MatrixXd::Zero(800, 800); // Store the color
    MatrixXd A = MatrixXd::Zero(800, 800); // Store the alpha mask

    const Vector3d camera_origin(0, 0, 3);
    const Vector3d camera_view_direction(0, 0, -1);

    // The camera is perspective, pointing in the direction -z and covering the unit square (-1,1) in x and y
    const Vector3d image_origin(-1, 1, 1);
    const Vector3d x_displacement(2.0 / A.cols(), 0, 0);
    const Vector3d y_displacement(0, -2.0 / A.rows(), 0);

    // Sphere setup
    const Vector3d sphere_center(0, 0, 0); // c
    const double sphere_radius = 0.9;      // R

    // material params
    const Vector3d diffuse_color(1, 0, 1);
    const double specular_exponent = 100;
    const Vector3d specular_color(0., 0, 1);

    // Single light source
    const Vector3d light_position(-1, 1, 1);
    double ambient = 0.1;

    for (unsigned i = 0; i < C.cols(); ++i)
    {
        for (unsigned j = 0; j < C.rows(); ++j)
        {
            const Vector3d pixel_center = image_origin + double(i) * x_displacement + double(j) * y_displacement;

            // Prepare the ray for perspective
            // ray is starting from the camera

            // Ex.1 Basic Ray Tracing - 6.
            const Vector3d ray_origin = camera_origin; // e
            // From camera origin to pixel
            const Vector3d ray_direction = pixel_center - camera_origin; // d

            // Intersect with the sphere
            // Ex.1 Basic Ray Tracing - 5.
            if (raysphere(sphere_center, ray_origin, ray_direction, sphere_radius))
            {
                // The ray hit the sphere, compute the exact intersection point
                Vector3d ray_intersection = getPointSphere(sphere_center, ray_origin, ray_direction, sphere_radius);

                // Compute normal at the intersection point
                // Vector3d ray_normal = ray_intersection.normalized();
                Vector3d ray_normal = ((sphere_center - ray_intersection) / sphere_radius).normalized();

                // TODO: Add shading parameter here
                // const double diffuse = (light_position - ray_intersection).normalized().dot(ray_normal);
                // const double specular = (light_position - ray_intersection).normalized().dot(ray_normal);

                Vector3d l = (ray_intersection - light_position).normalized();
                Vector3d v = (ray_intersection - ray_origin).normalized();
                Vector3d h = ((v + l) / ((v + l).norm())).normalized();

                Vector3d diffuse = diffuse_color * 1 * fmax(0, (ray_normal.dot(l)));
                Vector3d specular = specular_color * 1 * pow(fmax(0, (ray_normal.dot(h))), specular_exponent);

                // Simple diffuse model
                // C(i, j) = ambient + diffuse + specular;
                R(i, j) = ambient + diffuse(0) + specular(0);
                G(i, j) = ambient + diffuse(1) + specular(1);
                B(i, j) = ambient + diffuse(2) + specular(2);

                // Clamp to zero
                // C(i, j) = std::max(C(i, j), 0.);
                R(i, j) = std::max(R(i, j), 0.);
                G(i, j) = std::max(G(i, j), 0.);
                B(i, j) = std::max(B(i, j), 0.);

                // Disable the alpha mask for this pixel
                A(i, j) = 1;
            }
        }
    }

    // Save to png
    write_matrix_to_png(R, G, B, A, filename);
}

int main()
{
    raytrace_sphere();
    raytrace_parallelogram();
    raytrace_perspective();
    raytrace_shading();

    return 0;
}
