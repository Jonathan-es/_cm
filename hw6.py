import math

class Point:
    """
    Represents a point in 2D Cartesian space (x, y).
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance_to(self, other):
        """Calculates Euclidean distance between two points: sqrt((x2-x1)^2 + (y2-y1)^2)"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def translate(self, dx, dy):
        """Moves the point by dx, dy."""
        self.x += dx
        self.y += dy

    def scale(self, factor):
        """Scales the point coordinates relative to the origin (0,0)."""
        self.x *= factor
        self.y *= factor

    def rotate(self, angle_degrees, origin=(0, 0)):
        """
        Rotates the point around a specific origin.
        Math:
        x' = (x-ox)cos(theta) - (y-oy)sin(theta) + ox
        y' = (x-ox)sin(theta) + (y-oy)cos(theta) + oy
        """
        rad = math.radians(angle_degrees)
        ox, oy = origin
        
        px = self.x - ox
        py = self.y - oy

        new_x = px * math.cos(rad) - py * math.sin(rad)
        new_y = px * math.sin(rad) + py * math.cos(rad)

        self.x = new_x + ox
        self.y = new_y + oy

    def __repr__(self):
        return f"Point({round(self.x, 2)}, {round(self.y, 2)})"


class Line:
    """
    Represents a line defined by two points P1 and P2.
    """
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    @property
    def length(self):
        return self.p1.distance_to(self.p2)

    def get_coefficients(self):
        """
        Returns A, B, C for the standard form: Ax + By = C
        derived from two points (x1, y1) and (x2, y2).
        A = y1 - y2
        B = x2 - x1
        C = Ax1 + By1
        """
        A = self.p1.y - self.p2.y
        B = self.p2.x - self.p1.x
        C = A * self.p1.x + B * self.p1.y
        return A, B, C

    def translate(self, dx, dy):
        self.p1.translate(dx, dy)
        self.p2.translate(dx, dy)

    def scale(self, factor):
        self.p1.scale(factor)
        self.p2.scale(factor)

    def rotate(self, angle_degrees, origin=(0,0)):
        self.p1.rotate(angle_degrees, origin)
        self.p2.rotate(angle_degrees, origin)

    def __repr__(self):
        return f"Line({self.p1}, {self.p2})"


class Circle:
    """
    Represents a circle defined by a center Point and a radius.
    """
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def translate(self, dx, dy):
        self.center.translate(dx, dy)

    def scale(self, factor):
        """Scales the circle's position (center) and its radius."""
        self.center.scale(factor)
        self.radius *= abs(factor)

    def rotate(self, angle_degrees, origin=(0,0)):
        """Rotates the center of the circle around an origin."""
        self.center.rotate(angle_degrees, origin)

    def __repr__(self):
        return f"Circle(Center={self.center}, R={round(self.radius, 2)})"


class Triangle:
    """
    Represents a triangle defined by three Points.
    """
    def __init__(self, p1, p2, p3):
        self.vertices = [p1, p2, p3]

    def translate(self, dx, dy):
        for v in self.vertices:
            v.translate(dx, dy)

    def scale(self, factor):
        for v in self.vertices:
            v.scale(factor)

    def rotate(self, angle_degrees, origin=(0,0)):
        for v in self.vertices:
            v.rotate(angle_degrees, origin)

    def get_side_lengths(self):
        d1 = self.vertices[0].distance_to(self.vertices[1])
        d2 = self.vertices[1].distance_to(self.vertices[2])
        d3 = self.vertices[2].distance_to(self.vertices[0])
        return sorted([d1, d2, d3])

    def __repr__(self):
        return f"Triangle({self.vertices[0]}, {self.vertices[1]}, {self.vertices[2]})"


# ---------------------------------------------------------
# 1. Intersection Calculations
# ---------------------------------------------------------

def intersect_line_line(line1, line2):
    """
    Calculates intersection of two lines using Cramer's Rule on standard form.
    Line 1: A1x + B1y = C1
    Line 2: A2x + B2y = C2
    
    Determinant (det) = A1*B2 - A2*B1
    If det == 0, lines are parallel.
    x = (C1*B2 - C2*B1) / det
    y = (A1*C2 - A2*C1) / det
    """
    A1, B1, C1 = line1.get_coefficients()
    A2, B2, C2 = line2.get_coefficients()

    det = A1 * B2 - A2 * B1

    if abs(det) < 1e-9:
        return None  # Parallel lines

    x = (B2 * C1 - B1 * C2) / det
    y = (A1 * C2 - A2 * C1) / det
    return Point(x, y)


def intersect_line_circle(line, circle):
    """
    Finds intersection points between a line and a circle.
    Theory:
    1. Translate system so circle center is at (0,0).
    2. Line equation becomes parametric: P = P1 + t(P2 - P1).
    3. Substitute into x^2 + y^2 = r^2.
    4. Solve quadratic equation at^2 + bt + c = 0 for t.
    5. If discriminant < 0: No intersection.
       If discriminant = 0: Tangent (1 point).
       If discriminant > 0: Secant (2 points).
    """
    # Vector d = P2 - P1
    dx = line.p2.x - line.p1.x
    dy = line.p2.y - line.p1.y

    # Vector f = P1 - CircleCenter
    fx = line.p1.x - circle.center.x
    fy = line.p1.y - circle.center.y

    a = dx**2 + dy**2
    b = 2 * (fx * dx + fy * dy)
    c = (fx**2 + fy**2) - circle.radius**2

    discriminant = b**2 - 4*a*c

    if discriminant < 0:
        return []
    
    discriminant = math.sqrt(discriminant)
    t1 = (-b - discriminant) / (2*a)
    t2 = (-b + discriminant) / (2*a)

    points = []
    # Point 1
    points.append(Point(line.p1.x + t1*dx, line.p1.y + t1*dy))
    
    if discriminant > 1e-9: # If not tangent, add second point
        points.append(Point(line.p1.x + t2*dx, line.p1.y + t2*dy))
        
    return points


def intersect_circle_circle(c1, c2):
    """
    Finds intersections of two circles.
    Theory:
    Calculates distance (d) between centers.
    If d > r1+r2: Separate.
    If d < |r1-r2|: One inside other.
    If d = 0: Concentric.
    Otherwise, uses radical axis theorem to find chord of intersection.
    """
    d = c1.center.distance_to(c2.center)

    # Check for non-intersecting cases
    if d > c1.radius + c2.radius or d < abs(c1.radius - c2.radius) or d == 0:
        return []

    # 'a' is distance from c1 center to the perpendicular line connecting intersections
    a = (c1.radius**2 - c2.radius**2 + d**2) / (2 * d)
    
    # 'h' is the distance from that line to the intersection points
    h = math.sqrt(max(0, c1.radius**2 - a**2))

    # Calculate point P2 (projection of intersection on the line connecting centers)
    p2_x = c1.center.x + a * (c2.center.x - c1.center.x) / d
    p2_y = c1.center.y + a * (c2.center.y - c1.center.y) / d

    # Offset from P2 to find intersection points
    x3_1 = p2_x + h * (c2.center.y - c1.center.y) / d
    y3_1 = p2_y - h * (c2.center.x - c1.center.x) / d
    
    x3_2 = p2_x - h * (c2.center.y - c1.center.y) / d
    y3_2 = p2_y + h * (c2.center.x - c1.center.x) / d

    return [Point(x3_1, y3_1), Point(x3_2, y3_2)]


# ---------------------------------------------------------
# 2. Perpendicular Line Construction
# ---------------------------------------------------------

def get_perpendicular_line(line, point):
    """
    Constructs a perpendicular line from an external point to a given line.
    
    Theory:
    1. Find slope of given line (m).
    2. Slope of perpendicular is -1/m.
    3. We need the "foot" of the perpendicular (projection of point onto line).
       This can be found by intersecting the original line with a new line 
       defined by the external point and the perpendicular slope.
    """
    A, B, C = line.get_coefficients()
    
    # Perpendicular line has equation -Bx + Ay = C_perp
    C_perp = -B * point.x + A * point.y
    
    # Solve system:
    # 1) Ax + By = C
    # 2) -Bx + Ay = C_perp
    
    det = A*A - (-B)*B # A^2 + B^2
    
    if det == 0: return None 
    
    foot_x = (C*A - B*C_perp) / det
    foot_y = (C*B + A*C_perp) / det
    
    foot_point = Point(foot_x, foot_y)
    
    # Return a Line object from external point to the foot
    return Line(point, foot_point)


# ---------------------------------------------------------
# 3. Pythagorean Theorem Verification
# ---------------------------------------------------------

def verify_pythagorean(line, external_point):
    """
    Verifies Pythagorean theorem on a triangle formed by:
    1. External Point (A)
    2. Foot of perpendicular on line (B)
    3. An arbitrary third point on the line (C)
    
    Theorem: AB^2 + BC^2 = AC^2 (Hypotenuse)
    """
    # 1. Get the perpendicular line to find the foot (Point B)
    perp_line = get_perpendicular_line(line, external_point)
    foot_point = perp_line.p2 # Based on implementation above, p2 is the foot
    
    # 2. Pick a third point (C) on the line. 
    # We can just use line.p1 or line.p2, provided it's not the foot itself.
    if foot_point.distance_to(line.p1) > 1e-5:
        third_point = line.p1
    else:
        third_point = line.p2
        
    # 3. Create Triangle
    tri = Triangle(external_point, foot_point, third_point)
    
    # 4. Calculate squared lengths
    # Leg 1: External to Foot
    a_sq = external_point.distance_to(foot_point)**2
    # Leg 2: Foot to Third Point
    b_sq = foot_point.distance_to(third_point)**2
    # Hypotenuse: External to Third Point
    c_sq = external_point.distance_to(third_point)**2
    
    # Check equality (with float tolerance)
    is_valid = math.isclose(a_sq + b_sq, c_sq, rel_tol=1e-9)
    
    return {
        "valid": is_valid,
        "a_squared": a_sq,
        "b_squared": b_sq,
        "sum_legs": a_sq + b_sq,
        "hyp_squared": c_sq
    }


# ---------------------------------------------------------
# Main Execution / Examples
# ---------------------------------------------------------

if __name__ == "__main__":
    print("--- Geometric Library Demo ---\n")

    # 1. Line-Line Intersection
    l1 = Line(Point(0, 0), Point(4, 4))   # y = x
    l2 = Line(Point(0, 4), Point(4, 0))   # y = -x + 4
    inter_ll = intersect_line_line(l1, l2)
    print(f"1. Intersection of {l1} and {l2}: {inter_ll}")

    # 2. Line-Circle Intersection
    circ = Circle(Point(0, 0), 5)
    line_cut = Line(Point(-10, 3), Point(10, 3)) # Horizontal line at y=3
    inter_lc = intersect_line_circle(line_cut, circ)
    print(f"2. Intersection of {circ} and {line_cut}: {inter_lc}")

    # 3. Perpendicular Line
    base_line = Line(Point(0, 0), Point(10, 0)) # x-axis
    pt_outside = Point(5, 5)
    perp_line = get_perpendicular_line(base_line, pt_outside)
    print(f"3. Perpendicular from {pt_outside} to {base_line} goes to {perp_line.p2}")

    # 4. Pythagorean Verification
    print("4. Pythagorean Verification:")
    pythag_result = verify_pythagorean(base_line, pt_outside)
    print(f"   Is Right Triangle? {pythag_result['valid']}")
    print(f"   a^2 + b^2 = {round(pythag_result['sum_legs'], 2)}")
    print(f"   c^2       = {round(pythag_result['hyp_squared'], 2)}")

    # 5. Transformations (Rotation)
    print("5. Transformations:")
    tri = Triangle(Point(0, 0), Point(4, 0), Point(0, 3)) # 3-4-5 Triangle
    print(f"   Original Triangle: {tri}")
    
    # Rotate 90 degrees around origin
    tri.rotate(90)
    print(f"   Rotated 90 deg:    {tri}")
    
    # Scale by 2
    tri.scale(2)
    print(f"   Scaled by 2:       {tri}")