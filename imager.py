# Fusion 360 API Python
# Save 10 screenshots: 6 faces + 4 top-isometric corners
# Author: ChatGPT
import adsk.core, adsk.fusion, adsk.cam
import traceback, os, math, datetime
def _iter_all_components(root: adsk.fusion.Component):
    """Yield root and all child components via occurrences."""
    yield root
    for occ in root.occurrences:
        yield occ.component

def label_sketch_profiles_custom_graphics(
    text_height,
    font: str = "Arial",
    group_name: str = "SketchProfileLabelsCG"
):
    """
    Create custom-graphics text for every profile of every sketch in the design.
    Label format: S{sketchIndex}P{profileIndex}.
    Text is oriented in model space so local Z = sketch plane normal.
    Appends (gtext, base_pt, normal) to the global labels2.
    Returns (cg_group, created_count).
    """
    global labels2

    app = adsk.core.Application.get()
    design = adsk.fusion.Design.cast(app.activeProduct)
    if not design:
        raise RuntimeError("No active Fusion design.")
    root = design.rootComponent

    # Create/replace a dedicated CG group for sketch profile labels
    groups = root.customGraphicsGroups
    existing = None
    for i in range(groups.count):
        g = groups.item(i)
        try:
            if g.name == group_name:
                existing = g
                break
        except:
            pass
    if existing:
        existing.deleteMe()
    cg = groups.add()
    try:
        cg.name = group_name
    except:
        pass

    created = 0
    s_idx = 0
    ui.messageBox("call to prof")
    for comp in _iter_all_components(root):
        for sk in comp.sketches:
            ui.messageBox(str(sk))
            # Sketch plane normal in model coords
            try:
                n = sk.referencePlane.geometry.normal  # or sk.sketchPlane.geometry.normal (older API)
            except:
                try:
                    n = sk.sketchPlane.geometry.normal
                except:
                    n = adsk.core.Vector3D.create(0, 0, 1)
            if n.length < 1e-12:
                n = adsk.core.Vector3D.create(0, 0, 1)
            else:
                n.normalize()

            # Label each profile in this sketch
            for p_idx in range(sk.profiles.count):
                prof = sk.profiles.item(p_idx)

                # Robust centroid in model space
                pt3d = None
                try:
                    ap = prof.areaProperties()  # returns AreaProperties
                    c = ap.centroid  # Point3D in model space in recent API builds
                    # Some builds may return Point2D; convert if needed
                    if hasattr(c, "z"):
                        pt3d = c  # already Point3D
                    else:
                        # Treat as Point2D -> lift to sketch space then to model space
                        pt3d = adsk.core.Point3D.create(c.x, c.y, 0.0)
                        pt3d = sk.sketchToModelSpace(pt3d)
                except:
                    # Fallback: use sketch origin projected (not ideal, but avoids crashes)
                    try:
                        pt3d = sk.referencePlane.geometry.origin
                    except:
                        pt3d = adsk.core.Point3D.create(0, 0, 0)

                # Build orientation basis from the sketch normal
                x, y, z = _basis_from_normal(n)
                m = adsk.core.Matrix3D.create()
                m.setWithCoordinateSystem(pt3d, x, y, z)
                _ui.messageBox(pstr(n))
                label = f"S{s_idx}P{p_idx}"
                gtext = cg.addText(label, font, text_height, m)
                try:
                    gtext.depthPriority = 10
                except:
                    pass

                # Append to global so your nudge funcs can move them
                labels2.append((gtext, pt3d, n))
                created += 1

            s_idx += 1  # next sketch index

    app.activeViewport.refresh()
    return cg, created
# ---------- User settings ----------
IMAGE_WIDTH  = 336
IMAGE_HEIGHT = 336
USE_PERSPECTIVE = False  # set True if you prefer perspective shots
FILE_PREFIX = 'view'     # file names like view_01_front.png
# ----------------------------------
def pstr(x):
    return "("+str(x.x) + ", " + str(x.y) + ", " + str(x.z)+")"
cg_group, labels2 = None, None
_app = None
_ui = None
# globals
labels2 = []  # list[tuple[adsk.fusion.CustomGraphicsText, adsk.core.Point3D, adsk.core.Vector3D]]
def _view_dir_and_span():
    """Helper: returns (unit view_dir, eye_to_target_dist, model_span)."""
    app = adsk.core.Application.get()
    design = adsk.fusion.Design.cast(app.activeProduct)
    vp = app.activeViewport
    cam = vp.camera

    # unit view dir: camera -> target
    v = adsk.core.Vector3D.create(
        cam.target.x - cam.eye.x,
        cam.target.y - cam.eye.y,
        cam.target.z - cam.eye.z
    )
    if v.length < 1e-12:
        # degenerate, pick +Z
        v = adsk.core.Vector3D.create(0, 0, 1)
    else:
        v.normalize()

    eye_to_target = adsk.core.Vector3D.create(
        cam.target.x - cam.eye.x,
        cam.target.y - cam.eye.y,
        cam.target.z - cam.eye.z
    ).length

    rc = design.rootComponent
    bb = rc.boundingBox
    span = max(bb.maxPoint.x - bb.minPoint.x,
               bb.maxPoint.y - bb.minPoint.y,
               bb.maxPoint.z - bb.minPoint.z)
    span = max(span, 1e-9)
    return v, eye_to_target, span



def nudge_labels_towards_camera(offset_factor: float = 0.03,
                                min_eye_frac: float = 0.01):
    """
    Move labels from their stored BASE points toward the camera by a fixed amount.
    - offset_factor: fraction of model span
    - min_eye_frac:  fraction of eye->target distance (ensures visible movement when zoomed out)
    """
    global labels2
    if not labels2:
        return
    app = adsk.core.Application.get()
    vp = app.activeViewport

    v, eye_to_target, span = _view_dir_and_span()
    # toward camera is -view_dir
    step = max(span * offset_factor, eye_to_target * min_eye_frac)
    dx = -v.x * step
    dy = -v.y * step
    dz = -v.z * step

    for gtext, base_pt, _n in labels2:
        new_pt = adsk.core.Point3D.create(base_pt.x + dx,
                                          base_pt.y + dy,
                                          base_pt.z + dz)
        _place_label_at(gtext, new_pt)



def nudge_labels_away_from_camera(offset_factor: float = 0.03,
                                  min_eye_frac: float = 0.01):
    """
    Exact opposite of nudge_labels_towards_camera: uses the same base points
    and pushes labels away from the camera by the same amount.
    """
    global labels2
    if not labels2:
        return
    app = adsk.core.Application.get()
    vp = app.activeViewport

    v, eye_to_target, span = _view_dir_and_span()
    # away from camera is +view_dir
    step = max(span * offset_factor, eye_to_target * min_eye_frac)
    dx = v.x * step
    dy = v.y * step
    dz = v.z * step

    for gtext, base_pt, _n in labels2:
        new_pt = adsk.core.Point3D.create(base_pt.x + dx,
                                          base_pt.y + dy,
                                          base_pt.z + dz)
        _place_label_at(gtext, new_pt)

def face_midpoint_and_normal(face: adsk.fusion.BRepFace):
    """
    Returns (pt, n) where pt is the 3D point at the middle of the face's
    param domain and n is the unit normal there (fallback if unavailable).
    """
    ev = face.evaluator

    # Get parameter extents: [uMin, uMax], [vMin, vMax]

    # Evaluate point at (uMid, vMid)
    face_range = face.evaluator.parametricRange()
    midx = (face_range.maxPoint.x + face_range.minPoint.x)/2
    midy = (face_range.maxPoint.x + face_range.minPoint.x)/2
    okP, p = ev.getPointAtParameter(adsk.core.Point2D.create(midx,midy
        ))



    # Normal at (uMid, vMid) if available
    n = None

    okN, n = ev.getNormalAtPoint(p)

    n.normalize()

    return p, n
def _basis_from_normal(n: adsk.core.Vector3D):
    """Return orthonormal X,Y,Z where Z aligns with n and Y is upright."""
    z = n.copy()
    if z.length < 1e-12:
        z = adsk.core.Vector3D.create(0, 0, 1)
    else:
        z.normalize()

    helper = adsk.core.Vector3D.create(0, 0, 1)
    if abs(z.dotProduct(helper)) > 0.95:
        helper = adsk.core.Vector3D.create(0, 1, 0)

    x = helper.crossProduct(z); x.normalize()
    y = z.crossProduct(x);      y.normalize()

    # keep text upright (not mirrored) relative to world +Z
    if y.dotProduct(adsk.core.Vector3D.create(0,0,1)) < 0:
        x.scaleBy(-1); y.scaleBy(-1)
    return x, y, z
def label_brep_faces_custom_graphics(
    text_height,
    prefix: str = ["B","F"],
    font: str = "Arial",
    group_name: str = "FaceLabelsCG"
):
    global labels2
    app = adsk.core.Application.get()
    ui  = app.userInterface
    design = adsk.fusion.Design.cast(app.activeProduct)
    if not design:
        raise RuntimeError("No active Fusion design.")

    ui.activeSelections.clear()
    root = design.rootComponent

    # Recreate group
    cg_groups = root.customGraphicsGroups
    existing = None
    for i in range(cg_groups.count):
        g = cg_groups.item(i)
        try:
            if g.name == group_name:
                existing = g
                break
        except:
            pass
    if existing:
        existing.deleteMe()
    cg = cg_groups.add()
    try: cg.name = group_name
    except: pass

    def _all_bodies(comp: adsk.fusion.Component):
        for b in comp.bRepBodies:
            if b.isSolid or b.isSurface:
                yield b
        for occ in comp.occurrences:
            for b in occ.component.bRepBodies:
                if b.isSolid or b.isSurface:
                    yield b

    def _bbox_center(bb: adsk.core.OrientedBoundingBox3D):
        mp, Mp = bb.minPoint, bb.maxPoint
        return adsk.core.Point3D.create(
            0.5*(mp.x+Mp.x), 0.5*(mp.y+Mp.y), 0.5*(mp.z+Mp.z)
        )

    labels2 = []  # (gtext, anchor Point3D, outward normal)
    face_index = 0
    body_index = 0
    for body in _all_bodies(root):
        body_center = _bbox_center(body.boundingBox)
        face_index = 0
        for face in body.faces:
            label = f"{prefix[0]}{body_index}{prefix[1]}{face_index}"
            pt,n = face_midpoint_and_normal(face) 

            # flip normal to point roughly outward from the body's center
            out_vec = adsk.core.Vector3D.create(pt.x - body_center.x,
                                                pt.y - body_center.y,
                                                pt.z - body_center.z)
            if out_vec.length > 1e-12 and n.dotProduct(out_vec) < 0:
                n.scaleBy(-1.0)

            # create text at the base point
            x, y, z = _basis_from_normal(n)

            m = adsk.core.Matrix3D.create()
            m.setWithCoordinateSystem(pt, x, y, z)   # origin=pt, axes=(x,y,z)
            gtext = cg.addText(label, font, text_height, m)

            # billboard so it faces screen



            try: gtext.depthPriority = 10
            except: pass

            labels2.append((gtext, pt, n))
            face_index += 1
        body_index += 1
    app.activeViewport.refresh()
    return cg, labels2

def _model_span_and_center(design: adsk.fusion.Design):
    rc = design.rootComponent
    bb = rc.boundingBox
    minP, maxP = bb.minPoint, bb.maxPoint
    sx = maxP.x - minP.x
    sy = maxP.y - minP.y
    sz = maxP.z - minP.z
    span = max(sx, sy, sz)
    cx = 0.5*(minP.x + maxP.x)
    cy = 0.5*(minP.y + maxP.y)
    cz = 0.5*(minP.z + minP.z)  # center Z is not used; minor typo-safe
    return span, adsk.core.Point3D.create(cx, cy, cz)
def nudge_labels_off_surface(offset_factor=0.02, min_step_abs=None):
    """
    Push each label outward along its stored face normal.
    offset_factor: fraction of model max span to offset (e.g., 0.02 = 2%)
    min_step_abs: optional absolute fallback distance (in doc units)
    """
    global labels2
    if not labels2:
        return

    app = adsk.core.Application.get()
    design = adsk.fusion.Design.cast(app.activeProduct)
    if not design:
        return
    vp = app.activeViewport

    # model span as a scale reference
    rc = design.rootComponent
    bb = rc.boundingBox
    span = max(bb.maxPoint.x - bb.minPoint.x,
               bb.maxPoint.y - bb.minPoint.y,
               bb.maxPoint.z - bb.minPoint.z)
    span = max(span, 1e-9)
    step = max(span * offset_factor, (min_step_abs or 0.0))

    for gtext, base_pt, n in labels2:
        # safety: normalize n
        if n.length < 1e-12:
            continue
        nn = n.copy(); nn.normalize()

        new_pt = adsk.core.Point3D.create(
            base_pt.x + nn.x * step,
            base_pt.y + nn.y * step,
            base_pt.z + nn.z * step
        )

        # move via transform
        m = adsk.core.Matrix3D.create()
        m.setToIdentity()
        m.translation = adsk.core.Vector3D.create(new_pt.x, new_pt.y, new_pt.z)
        try:
            gtext.transform = m
        except:
            pass



def _norm(v: adsk.core.Vector3D) -> float:
    return math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)

def _normalized(v: adsk.core.Vector3D) -> adsk.core.Vector3D:
    n = _norm(v)
    if n < 1e-9:
        return adsk.core.Vector3D.create(0,0,1)
    return adsk.core.Vector3D.create(v.x/n, v.y/n, v.z/n)

def _is_near_parallel(a: adsk.core.Vector3D, b: adsk.core.Vector3D, cos_thresh=0.995) -> bool:
    a_u = _normalized(a); b_u = _normalized(b)
    dot = a_u.x*b_u.x + a_u.y*b_u.y + a_u.z*b_u.z
    return abs(dot) > cos_thresh

def _bbox_center_and_extent(design: adsk.fusion.Design):
    rc = design.rootComponent
    bb = rc.boundingBox
    minP, maxP = bb.minPoint, bb.maxPoint
    cx = 0.5*(minP.x + maxP.x)
    cy = 0.5*(minP.y + maxP.y)
    cz = 0.5*(minP.z + maxP.z)
    center = adsk.core.Point3D.create(cx, cy, cz)

    sx = max(1e-9, maxP.x - minP.x)
    sy = max(1e-9, maxP.y - minP.y)
    sz = max(1e-9, maxP.z - minP.z)
    diag = math.sqrt(sx*sx + sy*sy + sz*sz)
    max_span = max(sx, sy, sz)

    # A comfortable distance factor so the whole model fits without calling fit()
    radius = 0.5 * diag
    distance = 2.2 * radius  # tweak if you want tighter/looser framing
    return center, max_span, radius, distance

def _choose_up(dir_vec: adsk.core.Vector3D) -> adsk.core.Vector3D:
    # Prefer world Z as up; if nearly parallel, fall back to world Y.
    worldZ = adsk.core.Vector3D.create(0,0,1)
    worldY = adsk.core.Vector3D.create(0,1,0)
    if _is_near_parallel(dir_vec, worldZ):
        return worldY
    return worldZ

def _set_camera_and_capture(viewport: adsk.core.Viewport,
                            center: adsk.core.Point3D,
                            dir_vec: adsk.core.Vector3D,
                            max_span: float,
                            distance: float,
                            use_perspective: bool,
                            outfile: str):

    #label_brep_faces()

    _app.userInterface.activeSelections.clear()
    cam = viewport.camera

    cam.isFitView = False
    cam.isPerspective = use_perspective
    cam.isSmoothTransition = False
    
    d = _normalized(dir_vec)
    eye = adsk.core.Point3D.create(center.x + d.x*distance,
                                   center.y + d.y*distance,
                                   center.z + d.z*distance)
    cam.eye = eye
    cam.target = center

    up = _choose_up(d)
    cam.upVector = up

    # For orthographic, control scale explicitly so the part fits
    if not use_perspective:
        cam.orthographicScale = max_span * 1.7

    viewport.camera = cam
    viewport.fit()          # <-- this is the key line

    #nudge_labels_towards_camera(0.05)
    viewport.refresh()
    # Save image
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    viewport.saveAsImageFile(outfile, IMAGE_WIDTH, IMAGE_HEIGHT)
    #nudge_labels_away_from_camera(0.05)
    

def _output_folder() -> str:
    # Create a dated folder in user's Documents
    docs = os.path.join(os.path.expanduser('~'), 'Documents')
    stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    return os.path.join(docs, f'FusionScreenshots_{stamp}')

def _directions():
    V = adsk.core.Vector3D.create
    faces = [
        ('front',  V(0,  1,  0)),
        ('back',   V(0, -1,  0)),
        ('left',   V(-1, 0,  0)),
        ('right',  V(1,  0,  0)),
        ('top',    V(0,  0,  1)),
        ('bottom', V(0,  0, -1)),
    ]
    # Top-isometric corners (Z positive): TFR, TFL, TBR, TBL
    corners = [
        ('iso_top_front_right', V( 1,  1,  1)),
        ('iso_top_front_left',  V(-1,  1,  1)),
        ('iso_top_back_right',  V( 1, -1,  1)),
        ('iso_top_back_left',   V(-1, -1,  1)),
    ]
    return faces + corners

def run(context):
    global _app, _ui, labels2
    try:
        _app = adsk.core.Application.get()
        _ui = _app.userInterface
        design = adsk.fusion.Design.cast(_app.activeProduct)
        if not design:
            raise RuntimeError('No active Fusion design.')

        vp = _app.activeViewport
        center, max_span, radius, distance = _bbox_center_and_extent(design)

        outdir = _output_folder()
        dirs = _directions()
        span, cent = _model_span_and_center(design)
        cg_group, labels2 = label_brep_faces_custom_graphics(span/ 10, prefix=["B","F"])
    label_sketch_profiles_custom_graphics(
        span/ 10,
        font: str = "Arial",
        group_name: str = "SketchProfileLabelsCG"
    )
        for idx, (name, vec) in enumerate(dirs, start=1):
            filename = f'{FILE_PREFIX}_{idx:02d}_{name}.png'
            path = os.path.join(outdir, filename)
            vp.camera.isSmoothTransition = False
            _set_camera_and_capture(
                vp, center, vec, max_span, distance, USE_PERSPECTIVE, path
            )
            vp.camera.isSmoothTransition = True

        _ui.messageBox(f'Screenshots saved:\n{outdir}')

    except:
        if _ui:
            _ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))
        else:
            print('Failed:\n{}'.format(traceback.format_exc()))

def stop(context):
    # Nothing persistent to clean up.
    pass