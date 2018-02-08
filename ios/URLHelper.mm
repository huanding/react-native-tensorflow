#import "URLHelper.h"

#include "tensorflow/core/public/session.h"

#include <string>
#include <fstream>

@implementation URLHelper
{
}

+ (NSURL *) toURL: (NSString *) uri
{
    LOG(INFO) << "Attempting to load " << [uri UTF8String];

    NSURL * url = [NSURL URLWithString:uri];
    if (url && url.scheme && url.host) {
        LOG(INFO) << "Loading URL " << [[url absoluteString] UTF8String];
        return url;
    }

    if ([[NSFileManager defaultManager] fileExistsAtPath:uri]) {
        url = [NSURL fileURLWithPath:uri];
        if (url && url.scheme) {
            LOG(INFO) << "Loading file " << [[url absoluteString] UTF8String];
            return url;
        }
    }

    NSString * path = [[NSBundle mainBundle]
                       pathForResource:[[uri lastPathComponent] stringByDeletingPathExtension]
                       ofType:[uri pathExtension]];
    if (path) {
        url = [NSURL fileURLWithPath:path];
        if (url && url.scheme) {
            LOG(INFO) << "Loading resource " << [[url absoluteString] UTF8String];
            return url;
        }
    }

    throw std::invalid_argument([uri UTF8String]);
}
@end
