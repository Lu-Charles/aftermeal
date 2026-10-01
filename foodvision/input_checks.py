"""Deterministic input checks; visual assertions are explicitly user supplied."""
import hashlib
import json
POLICY_VERSION = 'input-checks-v2'
MIN_SIDE = 224
STARTING_PORTIONS = {'visible_food', 'empty_or_residue', 'uncertain'}

def pixel_digest(image):
    return hashlib.sha256(f'RGB:{image.width}x{image.height}:'.encode() + image.tobytes()).hexdigest()

def inspect_pair(images, starting_portion=None, flagged_images=()):
    if starting_portion is not None and starting_portion not in STARTING_PORTIONS:
        raise ValueError('Starting portion must be visible_food, empty_or_residue or uncertain.')
    issues = []
    digests = [pixel_digest(image) for image in images]
    for role, image in zip(('before', 'after'), images):
        if min(image.size) < MIN_SIDE:
            issues.append({'code': 'image_too_small', 'image': role, 'message': f'Use a {role} photo at least 224 pixels wide and tall.'})
        if all((low == high for low, high in image.getextrema())):
            issues.append({'code': 'constant_image', 'image': role, 'message': f'The {role} image has no visible variation. Choose a meal photograph.'})
    if digests[0] == digests[1]:
        issues.append({'code': 'duplicate_photos', 'image': 'pair', 'message': 'Both files contain the same image. Choose separate before and after photographs.'})
    status = 'rejected' if issues else 'ready'
    if not issues:
        matched = [f for f in flagged_images if f['decoded_rgb_sha256'] == digests[0]]
        if matched:
            issues.append({'code': 'known_starting_image_flag', 'image': 'before', 'message': 'This starting image matches an unresolved source-review flag. Use a verified starting photograph.', 'records': [f['record_id'] for f in matched], 'evidence': 'AI review proposal, not a verified empty-plate label'})
        if starting_portion == 'empty_or_residue':
            issues.append({'code': 'starting_portion_unverified', 'image': 'before', 'message': 'Choose a before photo with food on the plate.'})
        if issues:
            status = 'needs_review'
    return {'status': status, 'policy_version': POLICY_VERSION, 'issues': issues, 'starting_portion': {'value': starting_portion, 'source': 'user_observation' if starting_portion else 'not_provided'}, 'automatic_food_detection': False, 'decoded_image_sha256': dict(zip(('before', 'after'), digests))}

def load_flags(root):
    path = root / 'data/input_review_flags.json'
    return json.loads(path.read_text())['images']
