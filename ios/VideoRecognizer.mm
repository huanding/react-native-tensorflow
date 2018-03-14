#import "VideoRecognizer.h"
#import "URLHelper.h"
#import "ImageProcessor.h"

#include "tensorflow/core/public/session.h"

#import <AVFoundation/AVFoundation.h>

#include <fstream>
#include <string>

@implementation VideoRecognizer
{
    ImageProcessor * imageProcessor;
}

- (id) initWithData:(NSString *)modelUri labels:(NSString *)labelUri
{
    self = [super init];
    if (self != nil) {
    	imageProcessor = [[ImageProcessor alloc] initWithData:modelUri labels:labelUri];
    }
    return self;
}

- (NSArray *) recognize:(NSString *)videoUri maxResults:(NSNumber *)maxResults threshold:(NSNumber *)threshold
{
    NSURL * url = [URLHelper toURL:videoUri];
    AVURLAsset * asset = [[AVURLAsset alloc] initWithURL:url options:nil];
    if (asset == NULL) {
        throw std::invalid_argument("Failed to load video asset from url");
    }
    CGImagePropertyOrientation orientation = getOrientation(asset);
    CMTime duration = [asset duration];
    LOG(INFO) << "orientation: " << orientation << " duration: " << CMTimeGetSeconds(duration) << " seconds";

    NSMutableArray * timestamps = [[NSMutableArray alloc] init];
    for (int i = 0; i < CMTimeGetSeconds(duration); i++) {
      [timestamps addObject: [NSValue valueWithCMTime:CMTimeMakeWithSeconds(i, 30)]];
    }

    NSArray * results = [self recognizeImage:asset timestamps:timestamps orientation:orientation
        maxResults:maxResults threshold:threshold];
    return results;
}

- (NSArray *) recognizeImage:(AVAsset *)asset timestamps:(NSArray *)timestamps
                             orientation:(CGImagePropertyOrientation)orientation
                             maxResults:(NSNumber *)maxResults threshold:(NSNumber *)threshold
{
    AVAssetImageGenerator * gen = [AVAssetImageGenerator assetImageGeneratorWithAsset:asset];
    gen.requestedTimeToleranceAfter = kCMTimeZero;
    gen.requestedTimeToleranceBefore = kCMTimeZero;
    gen.appliesPreferredTrackTransform = YES;

    NSMutableArray * results = [[NSMutableArray alloc] init];

    // TODO: generateCGImagesAsynchronously for multiple thumbnail generation
    // TODO: run batch inference
    for (NSValue * timestamp in timestamps) {
        CMTime expectedTime = [timestamp CMTimeValue];
        LOG(INFO) << "Inferring image at " << CMTimeGetSeconds(expectedTime) << " seconds";
        CMTime actualTime;
        NSError * error = nil;
        CGImageRef image = [gen copyCGImageAtTime:expectedTime actualTime:&actualTime error:&error];
        LOG(INFO) << "Inferred image at " << CMTimeGetSeconds(actualTime) << " seconds";
        NSArray * result = [imageProcessor recognize:image orientation:orientation maxResults:maxResults threshold:threshold];
        [results addObject:result];
        [imageProcessor reset];
    }

    return results;
}

CGImagePropertyOrientation getOrientation(AVAsset * asset)
{
    AVAssetTrack * videoTrack = [[asset tracksWithMediaType:AVMediaTypeVideo] objectAtIndex:0];
    CGAffineTransform txf = [videoTrack preferredTransform];
    CGFloat videoAngleInDegree  = atan2(txf.b, txf.a) * 180 / M_PI;

    LOG(INFO) << "Transform degree:" << videoAngleInDegree;

    switch ((int)videoAngleInDegree) {
        case 0:
            return kCGImagePropertyOrientationRight;
        case 90:
            return kCGImagePropertyOrientationUp;
        case 180:
            return kCGImagePropertyOrientationLeft;
        case -90:
            return kCGImagePropertyOrientationDown;
            break;
        default:
            throw std::invalid_argument("Invalid image orientation angle");
    }
}

@end
