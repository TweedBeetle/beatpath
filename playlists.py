import xml.etree.ElementTree as et
import gzip
from urllib.parse import unquote
import argparse

master_db_location = 'C:\\Users\\chris\\AppData\\Roaming\\Pioneer\\rekordbox\\master.db'


def get_memcue(time):
    child = et.Element('POSITION_MARK')
    child.set('Name', '')
    child.set('Type', '0')
    child.set('Num', '-1')
    child.set('Start', normalize_time(time))
    return child


def get_hotcue(time, num):
    child = et.Element('POSITION_MARK')
    child.set('Name', '')
    child.set('Type', '0')
    child.set('Red', '40')
    child.set('Green', '226')
    child.set('Blue', '20')
    child.set('Num', str(num))
    child.set('Start', normalize_time(time))
    return child


# # rekordbox_tree = et.parse(args.rekordbox_file)
# rekordbox_tracks = rekordbox_tree.getroot().findall('./COLLECTION/TRACK')
#
#
# outfile = 'output.xml'
# print('Converting Ableton warp markers to Rekordbox cues.')
# for track in tracks:
#     filename = get_ableton_filename(track)
#     warp_markers = track.findall('.//WarpMarkers/WarpMarker')
#     # Find the corresponding track in rekordbox
#     for rekordbox_track in rekordbox_tracks:
#         if (get_rekordbox_filename(rekordbox_track) == filename):
#             print('processing ' + filename)
#             # clear all existing cues
#             for element in rekordbox_track.findall('./POSITION_MARK'):
#                 rekordbox_track.remove(element)
#             # create a hotcue and mem cue for each warp marker
#             num = 0
#             times = [float(marker.get('SecTime')) for marker in warp_markers]
#             times.sort()
#             # ignore last item cuz it gets duplicated for some reason
#             del times[-1]
#             for time in times:
#                 hotcue = get_hotcue(time, num)
#                 memcue = get_memcue(time)
#                 num = num + 1
#                 rekordbox_track.append(hotcue)
#                 rekordbox_track.append(memcue)
# rekordbox_tree.write(outfile, encoding='UTF-8', xml_declaration=True)

if __name__ == '__main__':
    pass
