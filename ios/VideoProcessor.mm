#import "VideoProcessor.h"
#include "URLHelper.h"

#import <AVFoundation/AVFoundation.h>

#include <string>
#include <fstream>

@implementation VideoProcessor
{
}

- (NSDictionary *) extractThumbnails:(NSString *)videoUri timestamps:(NSArray *)timestamps
{
    NSURL * url = [URLHelper toURL:videoUri];
    AVURLAsset * asset = [[AVURLAsset alloc] initWithURL:url options:nil];
    AVAssetImageGenerator *gen = [AVAssetImageGenerator assetImageGeneratorWithAsset:asset];
    gen.appliesPreferredTrackTransform = YES;

    NSMutableArray * cmtimes = [[NSMutableArray alloc] init];
    NSMutableArray * thumbnails = [[NSMutableArray alloc] init];

    // TODO: generateCGImagesAsynchronously for multiple thumbnail generation
    for (NSNumber * timestamp in timestamps) {
        CMTime expectedTime = CMTimeMakeWithSeconds([timestamp floatValue]  , 60);
        CMTime actualTime;
        NSError * error = nil;
        CGImageRef image = [gen copyCGImageAtTime:expectedTime actualTime:&actualTime error:&error];
        UIImage * thumbnail = [[UIImage alloc] initWithCGImage:image];
        CGImageRelease(image);

        [cmtimes addObject:[NSValue valueWithCMTime:actualTime]];
        [thumbnails addObject:thumbnail];
    }

    return [NSDictionary dictionaryWithObjects:thumbnails forKeys:timestamps];
}

@end
